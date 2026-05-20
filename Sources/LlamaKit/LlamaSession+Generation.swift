import Foundation
import LlamaSwift

extension LlamaSession {
    public enum GenerationError: Error, CustomStringConvertible {
        case promptTooLong(tokenCount: Int, batchCapacity: Int)
        case decodingFailed

        public var description: String {
            switch self {
            case .promptTooLong(let tokens, let capacity):
                return "Prompt has \(tokens) tokens, exceeding the session's batch capacity of \(capacity)"
            case .decodingFailed:
                return "llama.cpp failed to decode tokens during generation"
            }
        }
    }

    /// Generates text from `prompt`, streaming chunks of decoded UTF-8 as they
    /// are sampled.
    ///
    /// - Parameters:
    ///   - prompt: The input string. Use ``LlamaModel/Tokenizer/applyChatTemplate(_:addGenerationPrompt:template:)-(_:_:_:)``
    ///     to format multi-turn conversations correctly.
    ///   - sampler: Sampling strategy. Defaults to standard temperature
    ///     sampling; pass ``Sampler/greedy`` for deterministic output.
    ///   - maxTokens: Hard cap on the number of new tokens to generate.
    ///   - resetState: When `true` (default), clears the session's KV cache
    ///     so each call starts fresh. Pass `false` to keep state across
    ///     `generate` calls — useful for multi-turn flows where you append
    ///     new tokens to an existing context.
    ///
    /// The stream finishes early if the model emits an end-of-generation
    /// token. Cancelling the consuming task cancels the underlying decode loop.
    public nonisolated func generate(
        prompt: String,
        sampler: Sampler = .temperature(),
        maxTokens: Int = 256,
        resetState: Bool = true
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runGeneration(
                        prompt: prompt,
                        sampler: sampler,
                        maxTokens: maxTokens,
                        resetState: resetState,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Convenience wrapper that collects the full stream into a single string.
    public nonisolated func complete(
        prompt: String,
        sampler: Sampler = .temperature(),
        maxTokens: Int = 256,
        resetState: Bool = true
    ) async throws -> String {
        var result = ""
        for try await chunk in generate(
            prompt: prompt,
            sampler: sampler,
            maxTokens: maxTokens,
            resetState: resetState
        ) {
            result += chunk
        }
        return result
    }

    fileprivate func runGeneration(
        prompt: String,
        sampler: Sampler,
        maxTokens: Int,
        resetState: Bool,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async throws {
        if resetState, let memory = llama_get_memory(pointer) {
            llama_memory_clear(memory, true)
        }

        let tokenizer = model.tokenizer
        let promptTokens = try tokenizer.encode(prompt, addSpecial: true, parseSpecial: true)
        guard !promptTokens.isEmpty else { return }
        guard promptTokens.count <= Int(batchSize) else {
            throw GenerationError.promptTooLong(
                tokenCount: promptTokens.count,
                batchCapacity: Int(batchSize)
            )
        }

        var batch = llama_batch_init(Int32(batchSize), 0, 1)
        defer { llama_batch_free(batch) }

        batch.n_tokens = Int32(promptTokens.count)
        for i in 0..<promptTokens.count {
            batch.token[i] = promptTokens[i]
            batch.pos[i] = Int32(i)
            batch.n_seq_id[i] = 1
            if let seqIds = batch.seq_id, let seqId = seqIds[i] {
                seqId[0] = 0
            }
            batch.logits[i] = (i == promptTokens.count - 1) ? 1 : 0
        }

        guard llama_decode(pointer, batch) == 0 else {
            throw GenerationError.decodingFailed
        }

        let samplerInstance = sampler.instantiate()
        var detokenizer = StreamingDetokenizer()
        var position = Int32(promptTokens.count)

        for _ in 0..<maxTokens {
            try Task.checkCancellation()

            let nextToken = llama_sampler_sample(samplerInstance.pointer, pointer, -1)
            llama_sampler_accept(samplerInstance.pointer, nextToken)

            if tokenizer.isEndOfGeneration(nextToken) { break }

            let chunk = detokenizer.consume(tokenizer.tokenBytes(nextToken))
            if !chunk.isEmpty {
                continuation.yield(chunk)
            }

            batch.n_tokens = 1
            batch.token[0] = nextToken
            batch.pos[0] = position
            batch.n_seq_id[0] = 1
            if let seqIds = batch.seq_id, let seqId = seqIds[0] {
                seqId[0] = 0
            }
            batch.logits[0] = 1
            position += 1

            guard llama_decode(pointer, batch) == 0 else {
                throw GenerationError.decodingFailed
            }
        }

        let tail = detokenizer.flush()
        if !tail.isEmpty {
            continuation.yield(tail)
        }
    }
}
