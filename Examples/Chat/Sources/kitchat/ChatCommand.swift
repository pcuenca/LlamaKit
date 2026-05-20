import ArgumentParser
import Foundation
import LlamaKit

@main
struct ChatCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "kitchat",
        abstract: "Interactive chat REPL using LlamaKit."
    )

    @Option(name: .long, help: "Path to a local GGUF model file.")
    var model: String?

    @Option(
        name: .customLong("hf"),
        help: ArgumentHelp(
            "Hugging Face model id, optionally with a `:QUANT` suffix.",
            discussion: "Example: Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0. When :QUANT is omitted, defaults to Q4_K_M."
        )
    )
    var repo: String?

    @Option(name: .long, help: "System instructions prepended to every conversation.")
    var instructions: String?

    @Option(name: .long, help: "Sampling temperature. `0` selects deterministic greedy sampling.")
    var temperature: Float = 0.8

    @Option(name: .long, help: "Maximum tokens generated per turn.")
    var maxTokens: Int = 512

    @Option(name: .long, help: "Random seed. Omit for non-deterministic sampling.")
    var seed: UInt32?

    @Flag(help: "Print llama.cpp / ggml diagnostics to stderr.")
    var verbose: Bool = false

    private func loadModel() async throws -> LlamaModel {
        if let model {
            return try LlamaModel(contentsOf: URL(fileURLWithPath: model))
        }
        let (repoId, filename) = parseRepoSpec(repo!)
        return try await LlamaModel.from(repo: repoId, filename: filename)
    }

    private func parseRepoSpec(_ spec: String) -> (repo: String, filename: String) {
        if let colon = spec.firstIndex(of: ":") {
            let repoPart = String(spec[..<colon])
            let quantPart = String(spec[spec.index(after: colon)...])
            return (repoPart, "*\(quantPart).gguf")
        }
        return (spec, "*Q4_K_M.gguf")
    }

    func validate() throws {
        let hasLocal = model != nil
        let hasRemote = repo != nil
        guard hasLocal != hasRemote else {
            throw ValidationError("Provide exactly one of --model <path> or --hf <id>[:QUANT].")
        }
    }

    func run() async throws {
        LlamaBackend.loggingEnabled = verbose

        let model = try await loadModel()
        let session = try LlamaSession(model: model)
        let sampler: Sampler = temperature == 0
            ? .greedy
            : .temperature(temperature, seed: seed ?? .random(in: .min ... .max))

        var transcript: [ChatMessage] = []
        if let instructions {
            transcript.append(.system(instructions))
        }

        FileHandle.standardError.write(Data("\(model.modelDescription) loaded. /quit to exit, /reset to clear history.\n".utf8))

        while let line = prompt() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }

            switch trimmed {
            case "/quit", "/exit":
                return
            case "/reset":
                transcript.removeAll()
                if let instructions {
                    transcript.append(.system(instructions))
                }
                print("(conversation cleared)")
                continue
            default:
                break
            }

            transcript.append(.user(trimmed))
            let rendered = try model.tokenizer.applyChatTemplate(transcript)

            var assistant = ""
            do {
                for try await chunk in session.generate(
                    prompt: rendered,
                    sampler: sampler,
                    maxTokens: maxTokens
                ) {
                    print(chunk, terminator: "")
                    fflush(stdout)
                    assistant += chunk
                }
            } catch {
                print("\n(generation failed: \(error))")
                transcript.removeLast()
                continue
            }
            print()

            transcript.append(.assistant(assistant))
        }
    }

    private func prompt() -> String? {
        print("> ", terminator: "")
        fflush(stdout)
        return readLine()
    }
}
