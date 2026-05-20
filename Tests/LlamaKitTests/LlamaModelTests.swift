import Foundation
import Testing

@testable import LlamaKit

@Suite("LlamaModel")
struct LlamaModelTests {
    @Test
    func loadFailsForMissingFile() {
        let url = URL(fileURLWithPath: "/tmp/does-not-exist.gguf")
        #expect(throws: LlamaModel.LoadError.self) {
            _ = try LlamaModel(contentsOf: url)
        }
    }

    #if Hub
        /// Downloads a small GGUF and verifies the model's reported metadata.
        @Test
        func loadsSmolLM2FromHub() async throws {
            let model = try await LlamaModel.from(
                repo: "unsloth/SmolLM2-135M-Instruct-GGUF",
                filename: "*Q2_K.gguf"
            )

            #expect(model.parameterCount > 0)
            #expect(model.trainingContextLength > 0)
            #expect(model.embeddingDimension > 0)
            #expect(model.layerCount > 0)
            #expect(model.hasDecoder)

            let tokenizer = model.tokenizer
            #expect(tokenizer.vocabSize > 0)
            #expect(tokenizer.chatTemplate != nil)
            #expect(tokenizer.bosToken != nil)
            #expect(tokenizer.eosToken != nil)
            if let eos = tokenizer.eosToken {
                #expect(tokenizer.isEndOfGeneration(eos))
            }

            let session = try LlamaSession(model: model)
            #expect(session.contextLength > 0)
            #expect(session.batchSize > 0)
        }

        @Test
        func encodesAndDecodesText() async throws {
            let model = try await LlamaModel.from(
                repo: "unsloth/SmolLM2-135M-Instruct-GGUF",
                filename: "*Q2_K.gguf",
                parameters: .init(vocabularyOnly: true)
            )
            let tokenizer = model.tokenizer

            // Round-trip without special tokens
            let original = "Hello, world!"
            let tokens = try tokenizer.encode(original, addSpecial: false)
            #expect(!tokens.isEmpty)

            let decoded = tokenizer.decode(tokens)
            #expect(decoded == original)

            // Empty input round-trips to empty.
            #expect(try tokenizer.encode("").isEmpty)
            #expect(tokenizer.decode([]) == "")

            // addSpecial is permissive: it lets the tokenizer add BOS/EOS
            // *if the model's tokenizer config says to*. Many GGUFs declare
            // a BOS token but have `add_bos_token: false`, in which case
            // both calls return the same tokens.
            let withSpecial = try tokenizer.encode("Hi", addSpecial: true)
            let withoutSpecial = try tokenizer.encode("Hi", addSpecial: false)
            #expect(withSpecial.count >= withoutSpecial.count)
            if withSpecial.count > withoutSpecial.count, let bos = tokenizer.bosToken {
                #expect(withSpecial.first == bos)
            }

            // tokenToText on each piece concatenated should match decode()
            // (ignoring unicode for now; should work for complete sequences)
            let pieces = tokens.map { tokenizer.tokenToText($0) }.joined()
            #expect(pieces == original)
        }

        @Test
        func appliesChatTemplate() async throws {
            let model = try await LlamaModel.from(
                repo: "unsloth/SmolLM2-135M-Instruct-GGUF",
                filename: "*Q2_K.gguf",
                parameters: .init(vocabularyOnly: true)
            )
            let tokenizer = model.tokenizer

            let prompt = try tokenizer.applyChatTemplate([
                .system("You are a helpful assistant."),
                .user("Hello"),
            ])

            #expect(prompt.contains("You are a helpful assistant."))
            #expect(prompt.contains("Hello"))
            #expect(prompt.contains("<|im_start|>"))
            #expect(prompt.contains("assistant"))

            // Without addGenerationPrompt the prompt does not invite a reply.
            let closed = try tokenizer.applyChatTemplate(
                [.user("Hi")],
                addGenerationPrompt: false
            )
            let open = try tokenizer.applyChatTemplate(
                [.user("Hi")],
                addGenerationPrompt: true
            )
            #expect(open.count > closed.count)

            // Dict-shaped overload should render identically to the typed one,
            // and reject malformed entries.
            let viaDicts = try tokenizer.applyChatTemplate([
                ["role": "system", "content": "You are a helpful assistant."],
                ["role": "user", "content": "Hello"],
            ])
            #expect(viaDicts == prompt)

            #expect(throws: LlamaModel.Tokenizer.ChatTemplateError.self) {
                _ = try tokenizer.applyChatTemplate([["role": "user"]])
            }
        }
    #endif
}
