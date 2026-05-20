import Foundation
import Testing

@testable import LlamaKit

@Suite("LlamaSession")
struct LlamaSessionTests {
    #if Hub
        private func loadModel() async throws -> LlamaModel {
            try await LlamaModel.from(
                repo: "unsloth/SmolLM2-135M-Instruct-GGUF",
                filename: "*Q2_K.gguf"
            )
        }

        @Test
        func greedyCompletionReturnsText() async throws {
            let model = try await loadModel()
            let session = try LlamaSession(model: model)

            let prompt = try model.tokenizer.applyChatTemplate([
                .user("Say hello in one word.")
            ])

            let response = try await session.complete(
                prompt: prompt,
                sampler: .greedy,
                maxTokens: 32
            )

            #expect(!response.isEmpty)
        }

        @Test
        func greedyIsDeterministic() async throws {
            let model = try await loadModel()
            let session = try LlamaSession(model: model)

            let prompt = try model.tokenizer.applyChatTemplate([
                .user("Repeat the word: banana.")
            ])

            let first = try await session.complete(
                prompt: prompt,
                sampler: .greedy,
                maxTokens: 24
            )
            let second = try await session.complete(
                prompt: prompt,
                sampler: .greedy,
                maxTokens: 24
            )

            #expect(first == second)
        }

        @Test
        func streamYieldsIncrementalChunks() async throws {
            let model = try await loadModel()
            let session = try LlamaSession(model: model)

            let prompt = try model.tokenizer.applyChatTemplate([
                .user("Count from one to three.")
            ])

            var chunks: [String] = []
            for try await chunk in session.generate(
                prompt: prompt,
                sampler: .greedy,
                maxTokens: 24
            ) {
                chunks.append(chunk)
            }

            #expect(chunks.count >= 2)
            #expect(!chunks.joined().isEmpty)
        }

        @Test
        func maxTokensCapsOutput() async throws {
            let model = try await loadModel()
            let session = try LlamaSession(model: model)

            let prompt = try model.tokenizer.applyChatTemplate([
                .user("Write a long essay.")
            ])

            let response = try await session.complete(
                prompt: prompt,
                sampler: .greedy,
                maxTokens: 8
            )

            let tokenCount = try model.tokenizer.encode(response, addSpecial: false).count
            #expect(tokenCount <= 8)
        }
    #endif
}
