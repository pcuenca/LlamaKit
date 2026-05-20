import Foundation
import LlamaSwift

/// A stateful inference session bound to a `LlamaModel`.
///
/// A session owns the KV cache — the running record of every token the
/// model has processed. Each generation step extends that state, so a
/// single session represents one ongoing stream of work: a conversation,
/// an embedding batch, or an evaluation pass.
///
/// llama.cpp serializes calls into the underlying `llama_context`, so
/// `LlamaSession` is an `actor`. To run inference in parallel, create
/// multiple sessions from the same `LlamaModel` — they share weights but
/// each holds its own KV cache.
///
/// Wraps llama.cpp's `llama_context`.
public actor LlamaSession {
    /// Parameters that control how a session is initialized.
    public struct Parameters: Sendable {
        /// Maximum sequence length the session can hold, in tokens.
        ///
        /// `0` (default) uses the model's training context length.
        public var contextLength: UInt32

        /// Logical maximum batch size submitted in one step.
        public var batchSize: UInt32

        /// Physical maximum batch size.
        public var physicalBatchSize: UInt32

        /// Threads used for single-token generation.
        public var threadCount: Int32

        /// Threads used for prompt and batch processing.
        public var batchThreadCount: Int32

        /// Whether the session computes embeddings rather than generating text.
        public var embeddingMode: Bool

        public init(
            contextLength: UInt32 = 0,
            batchSize: UInt32 = 512,
            physicalBatchSize: UInt32 = 512,
            threadCount: Int32 = Int32(ProcessInfo.processInfo.processorCount),
            batchThreadCount: Int32 = Int32(ProcessInfo.processInfo.processorCount),
            embeddingMode: Bool = false
        ) {
            self.contextLength = contextLength
            self.batchSize = batchSize
            self.physicalBatchSize = physicalBatchSize
            self.threadCount = threadCount
            self.batchThreadCount = batchThreadCount
            self.embeddingMode = embeddingMode
        }

        public static let `default` = Parameters()

        fileprivate func toC() -> llama_context_params {
            var params = llama_context_default_params()
            params.n_ctx = contextLength
            params.n_batch = batchSize
            params.n_ubatch = physicalBatchSize
            params.n_threads = threadCount
            params.n_threads_batch = batchThreadCount
            params.embeddings = embeddingMode
            return params
        }
    }

    public enum InitError: Error, CustomStringConvertible {
        case initFailed
        case encoderOnlyModel

        public var description: String {
            switch self {
            case .initFailed:
                return "llama.cpp failed to initialize a session from the model"
            case .encoderOnlyModel:
                return "Model is encoder-only and cannot be used for generation"
            }
        }
    }

    nonisolated(unsafe) let pointer: OpaquePointer

    /// The model this session is bound to. We hold a strong reference to keep it alive.
    public nonisolated let model: LlamaModel

    /// The parameters used to initialize this session.
    public nonisolated let parameters: Parameters

    /// The active context length, as resolved by llama.cpp at init.
    ///
    /// Differs from `parameters.contextLength` when that was `0` (meaning
    /// "use the model's training context length").
    public nonisolated let contextLength: UInt32

    /// The active logical batch size.
    public nonisolated let batchSize: UInt32

    /// The active physical batch size.
    public nonisolated let physicalBatchSize: UInt32

    /// Creates a session for the given model.
    public init(model: LlamaModel, parameters: Parameters = .default) throws {
        let cParams = parameters.toC()
        guard let pointer = llama_init_from_model(model.pointer, cParams) else {
            throw InitError.initFailed
        }

        // Generation requires a KV cache. Embedding-only sessions and
        // encoder-only models won't have one.
        if !parameters.embeddingMode, llama_get_memory(pointer) == nil {
            llama_free(pointer)
            throw InitError.encoderOnlyModel
        }

        self.pointer = pointer
        self.model = model
        self.parameters = parameters
        self.contextLength = llama_n_ctx(pointer)
        self.batchSize = llama_n_batch(pointer)
        self.physicalBatchSize = llama_n_ubatch(pointer)
    }

    deinit {
        llama_free(pointer)
    }
}
