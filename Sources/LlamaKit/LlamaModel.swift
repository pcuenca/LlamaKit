import Foundation
import LlamaSwift

/// A loaded GGUF model.
///
/// `LlamaModel` owns the underlying `llama_model` pointer and frees it on
/// deinit. The model is immutable after loading and safe to share across
/// threads; mutable inference state lives in a separate context type.
public final class LlamaModel: @unchecked Sendable {
    /// Parameters that control how a model file is loaded.
    public struct Parameters: Sendable {
        /// Number of transformer layers to offload to Metal.
        ///
        /// `-1` (default) offloads all layers. Set to `0` to run on CPU.
        public var gpuLayerCount: Int32

        /// Memory-map the model file instead of copying it into RAM.
        public var useMemoryMap: Bool

        /// Lock the model in RAM so the OS can't page it out.
        public var useMemoryLock: Bool

        /// Load only the vocabulary, skipping weights.
        public var vocabularyOnly: Bool

        public init(
            gpuLayerCount: Int32 = -1,
            useMemoryMap: Bool = true,
            useMemoryLock: Bool = false,
            vocabularyOnly: Bool = false
        ) {
            self.gpuLayerCount = gpuLayerCount
            self.useMemoryMap = useMemoryMap
            self.useMemoryLock = useMemoryLock
            self.vocabularyOnly = vocabularyOnly
        }

        public static let `default` = Parameters()

        fileprivate func toC() -> llama_model_params {
            var params = llama_model_default_params()
            // llama.cpp uses INT32_MAX as the sentinel for "all layers".
            params.n_gpu_layers = gpuLayerCount < 0 ? Int32.max : gpuLayerCount
            params.use_mmap = useMemoryMap
            params.use_mlock = useMemoryLock
            params.vocab_only = vocabularyOnly
            return params
        }
    }

    public enum LoadError: Error, CustomStringConvertible {
        case fileNotFound(URL)
        case loadFailed(URL)

        public var description: String {
            switch self {
            case .fileNotFound(let url):
                return "GGUF model file not found at \(url.path)"
            case .loadFailed(let url):
                return "llama.cpp failed to load GGUF model at \(url.path)"
            }
        }
    }

    let pointer: OpaquePointer
    let vocabPointer: OpaquePointer

    /// The file the model was loaded from.
    public let url: URL

    /// The parameters used to load this model.
    public let parameters: Parameters

    /// Loads a GGUF model from a local file.
    public init(contentsOf url: URL, parameters: Parameters = .default) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadError.fileNotFound(url)
        }

        LlamaBackend.ensureInitialized()

        let cParams = parameters.toC()
        guard let pointer = llama_model_load_from_file(url.path, cParams) else {
            throw LoadError.loadFailed(url)
        }
        guard let vocab = llama_model_get_vocab(pointer) else {
            llama_model_free(pointer)
            throw LoadError.loadFailed(url)
        }

        self.pointer = pointer
        self.vocabPointer = vocab
        self.url = url
        self.parameters = parameters
    }

    deinit {
        llama_model_free(pointer)
    }
}

// MARK: - Metadata

extension LlamaModel {
    /// The maximum context length the model was trained with.
    public var trainingContextLength: Int32 {
        llama_model_n_ctx_train(pointer)
    }

    /// The embedding dimension.
    public var embeddingDimension: Int32 {
        llama_model_n_embd(pointer)
    }

    /// The number of transformer layers.
    public var layerCount: Int32 {
        llama_model_n_layer(pointer)
    }

    /// The number of attention heads.
    public var attentionHeadCount: Int32 {
        llama_model_n_head(pointer)
    }

    /// Whether the model has an encoder component (e.g. encoder-decoder
    /// architectures like T5).
    public var hasEncoder: Bool {
        llama_model_has_encoder(pointer)
    }

    /// Whether the model has a decoder component. Decoder-only LLMs return
    /// `true`; encoder-only models (e.g. BERT) return `false`.
    public var hasDecoder: Bool {
        llama_model_has_decoder(pointer)
    }

    /// A human-readable description like "llama 7B Q4_K - Medium".
    public var modelDescription: String {
        var buffer = [CChar](repeating: 0, count: 256)
        let written = llama_model_desc(pointer, &buffer, buffer.count)
        guard written > 0 else { return "" }
        return buffer.withUnsafeBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self).prefix(Int(written))
            return String(decoding: bytes, as: UTF8.self)
        }
    }

    /// The model's total parameter count.
    public var parameterCount: UInt64 {
        llama_model_n_params(pointer)
    }

    /// The on-disk size of the model in bytes.
    public var fileSize: UInt64 {
        llama_model_size(pointer)
    }
}

// MARK: - Tokenizer

extension LlamaModel {
    /// The model's tokenizer namespace, mirroring GGUF's `tokenizer.*` keys
    /// and Hugging Face's `tokenizer` convention.
    public var tokenizer: Tokenizer { Tokenizer(model: self) }

    /// A view over the model's vocabulary and tokenizer-level metadata.
    ///
    /// `Tokenizer` borrows from `LlamaModel` and holds a strong reference
    /// to keep the underlying pointers alive.
    public struct Tokenizer: Sendable {
        private let model: LlamaModel

        var vocab: OpaquePointer { model.vocabPointer }
        var modelPointer: OpaquePointer { model.pointer }

        init(model: LlamaModel) {
            self.model = model
        }

        /// The number of tokens in the vocabulary.
        public var size: Int32 {
            llama_vocab_n_tokens(vocab)
        }

        /// The chat template embedded in the GGUF file, if any.
        ///
        /// Returns `nil` when the file has no `tokenizer.chat_template`
        /// metadata. llama.cpp stores this at the model level but GGUF
        /// keeps it under `tokenizer.` on disk.
        public var chatTemplate: String? {
            guard let cString = llama_model_chat_template(modelPointer, nil) else {
                return nil
            }
            return String(cString: cString)
        }

        /// The beginning-of-sequence token, or `nil` if the model has none.
        public var bosToken: llama_token? {
            normalize(llama_vocab_bos(vocab))
        }

        /// The end-of-sequence token, or `nil` if the model has none.
        public var eosToken: llama_token? {
            normalize(llama_vocab_eos(vocab))
        }

        /// The end-of-turn token, or `nil` if the model has none.
        public var endOfTurnToken: llama_token? {
            normalize(llama_vocab_eot(vocab))
        }

        /// The sentence-separator token, or `nil` if the model has none.
        public var separatorToken: llama_token? {
            normalize(llama_vocab_sep(vocab))
        }

        /// The padding token, or `nil` if the model has none.
        public var paddingToken: llama_token? {
            normalize(llama_vocab_pad(vocab))
        }

        private func normalize(_ token: llama_token) -> llama_token? {
            token == LLAMA_TOKEN_NULL ? nil : token
        }
    }
}
