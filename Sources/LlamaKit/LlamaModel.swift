import Foundation
import Jinja
import LlamaSwift

/// A loaded GGUF model.
///
/// `LlamaModel` owns the underlying `llama_model` pointer and frees it on
/// deinit. The model is immutable after loading and safe to share across
/// threads; mutable inference state lives in a separate type (LlamaSession).
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
        public var vocabSize: Int32 {
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

        /// Whether `token` is an end-of-generation token (EOS, EOT, or any
        /// model-specific generation terminator).
        public func isEndOfGeneration(_ token: llama_token) -> Bool {
            llama_vocab_is_eog(vocab, token)
        }

        // MARK: - Encode / Decode

        public enum EncodingError: Error, CustomStringConvertible {
            case overflow

            public var description: String {
                switch self {
                case .overflow:
                    return "Tokenization result would exceed Int32.max tokens"
                }
            }
        }

        /// Tokenize `text` into a sequence of token ids.
        ///
        /// - Parameters:
        ///   - text: The input to tokenize.
        ///   - addSpecial: When `true`, allow the tokenizer to add BOS/EOS
        ///     tokens if the model is configured to do so. Default `true`;
        ///     pass `false` when tokenizing fragments that will be concatenated.
        ///   - parseSpecial: When `true`, parse special tokens like
        ///     `<|im_start|>` as themselves rather than as plain text.
        ///     Default `true` — what you want for prompt building.
        public func encode(
            _ text: String,
            addSpecial: Bool = true,
            parseSpecial: Bool = true
        ) throws -> [llama_token] {
            if text.isEmpty { return [] }

            let utf8Count = Int32(text.utf8.count)
            var capacity = max(utf8Count + 4, 16)
            var buffer = [llama_token](repeating: 0, count: Int(capacity))

            var written = buffer.withUnsafeMutableBufferPointer { ptr in
                llama_tokenize(
                    vocab, text, utf8Count,
                    ptr.baseAddress, capacity,
                    addSpecial, parseSpecial
                )
            }

            if written < 0 {
                if written == Int32.min { throw EncodingError.overflow }
                capacity = -written
                buffer = [llama_token](repeating: 0, count: Int(capacity))
                written = buffer.withUnsafeMutableBufferPointer { ptr in
                    llama_tokenize(
                        vocab, text, utf8Count,
                        ptr.baseAddress, capacity,
                        addSpecial, parseSpecial
                    )
                }
            }

            guard written >= 0 else { throw EncodingError.overflow }
            return Array(buffer.prefix(Int(written)))
        }

        /// Detokenize a sequence of tokens back into text.
        ///
        /// - Parameters:
        ///   - tokens: The tokens to render.
        ///   - removeSpecial: When `true`, strip BOS/EOS tokens from the output.
        ///   - renderSpecial: When `true`, render special tokens (e.g.
        ///     `<|im_start|>`) as literal text. Default `false` — what you
        ///     want for user-visible output.
        public func decode(
            _ tokens: [llama_token],
            removeSpecial: Bool = false,
            renderSpecial: Bool = false
        ) -> String {
            if tokens.isEmpty { return "" }

            var capacity = Int32(max(tokens.count * 8, 32))
            var buffer = [CChar](repeating: 0, count: Int(capacity))

            var written = tokens.withUnsafeBufferPointer { tokensPtr in
                buffer.withUnsafeMutableBufferPointer { bufPtr in
                    llama_detokenize(
                        vocab,
                        tokensPtr.baseAddress,
                        Int32(tokens.count),
                        bufPtr.baseAddress,
                        capacity,
                        removeSpecial,
                        renderSpecial
                    )
                }
            }

            if written < 0 {
                capacity = -written
                buffer = [CChar](repeating: 0, count: Int(capacity))
                written = tokens.withUnsafeBufferPointer { tokensPtr in
                    buffer.withUnsafeMutableBufferPointer { bufPtr in
                        llama_detokenize(
                            vocab,
                            tokensPtr.baseAddress,
                            Int32(tokens.count),
                            bufPtr.baseAddress,
                            capacity,
                            removeSpecial,
                            renderSpecial
                        )
                    }
                }
            }

            guard written > 0 else { return "" }
            return buffer.withUnsafeBytes { raw in
                let bytes = raw.bindMemory(to: UInt8.self).prefix(Int(written))
                return String(decoding: bytes, as: UTF8.self)
            }
        }

        // MARK: - Chat Templates

        public enum ChatTemplateError: Error, CustomStringConvertible {
            case noTemplate
            case applyFailed
            case malformedMessage

            public var description: String {
                switch self {
                case .noTemplate:
                    return "Model has no embedded chat template and no override was provided"
                case .applyFailed:
                    return "llama.cpp failed to apply the chat template"
                case .malformedMessage:
                    return "A message dictionary is missing the 'role' or 'content' key"
                }
            }
        }

        /// Render a chat into a prompt string using the model's embedded
        /// chat template.
        ///
        /// - Parameters:
        ///   - messages: The conversation, in order.
        ///   - addGenerationPrompt: When `true` (default), the rendered
        ///     output ends with the tokens that mark the start of an
        ///     assistant turn — ready for generation.
        ///   - template: Optional template override. Defaults to the
        ///     template embedded in the GGUF.
        ///
        /// - Note: llama.cpp does *not* run the stored template as Jinja.
        ///   It uses a built-in list of formatters indexed by template. The list may
        ///   be incomplete; llama.cpp may fall back to ChatML or fail.
        public func applyChatTemplate(
            _ messages: [ChatMessage],
            addGenerationPrompt: Bool = true,
            template: String? = nil
        ) throws -> String {
            guard let resolvedTemplate = template ?? chatTemplate else {
                throw ChatTemplateError.noTemplate
            }

            // Keep C strings alive for the duration of the call.
            let cRoles: [UnsafeMutablePointer<CChar>?] = messages.map { strdup($0.role) }
            let cContents: [UnsafeMutablePointer<CChar>?] = messages.map { strdup($0.content) }
            defer {
                for ptr in cRoles { free(ptr) }
                for ptr in cContents { free(ptr) }
            }

            let cMessages: [llama_chat_message] = zip(cRoles, cContents).map { role, content in
                llama_chat_message(role: role, content: content)
            }

            let requiredSize: Int32 = resolvedTemplate.withCString { tmpl in
                llama_chat_apply_template(
                    tmpl, cMessages, cMessages.count,
                    addGenerationPrompt, nil, 0
                )
            }

            // Use Jinja when the hardcoded template detection fails.
            // llama.cpp uses `common/jinja`, not bundled with the XCFramework.
            if requiredSize <= 0 {
                return try renderJinjaTemplate(
                    resolvedTemplate,
                    messages: messages,
                    addGenerationPrompt: addGenerationPrompt
                )
            }

            var buffer = [CChar](repeating: 0, count: Int(requiredSize) + 1)
            let written: Int32 = resolvedTemplate.withCString { tmpl in
                buffer.withUnsafeMutableBufferPointer { ptr in
                    llama_chat_apply_template(
                        tmpl, cMessages, cMessages.count,
                        addGenerationPrompt, ptr.baseAddress, Int32(ptr.count)
                    )
                }
            }
            guard written > 0 else { throw ChatTemplateError.applyFailed }

            return buffer.withUnsafeBytes { raw in
                let bytes = raw.bindMemory(to: UInt8.self).prefix(Int(written))
                return String(decoding: bytes, as: UTF8.self)
            }
        }

        private func renderJinjaTemplate(
            _ template: String,
            messages: [ChatMessage],
            addGenerationPrompt: Bool
        ) throws -> String {
            var context: [String: Jinja.Value] = [
                "messages": .array(messages.map { msg in
                    var dict = OrderedDictionary<String, Jinja.Value>()
                    dict["role"] = .string(msg.role)
                    dict["content"] = .string(msg.content)
                    return .object(dict)
                }),
                "add_generation_prompt": .boolean(addGenerationPrompt),
            ]
            if let bos = bosToken {
                context["bos_token"] = .string(tokenToText(bos, renderSpecial: true))
            }
            if let eos = eosToken {
                context["eos_token"] = .string(tokenToText(eos, renderSpecial: true))
            }
            do {
                // TODO: compiled template cache
                return try Jinja.Template(template).render(context)
            } catch {
                throw ChatTemplateError.applyFailed
            }
        }

        /// Render a chat into a prompt string from dictionary-shaped messages.
        ///
        /// Convenience overload for interop with Python-style message arrays.
        public func applyChatTemplate(
            _ messages: [[String: String]],
            addGenerationPrompt: Bool = true,
            template: String? = nil
        ) throws -> String {
            let typed: [ChatMessage] = try messages.map { dict in
                guard let role = dict["role"], let content = dict["content"] else {
                    throw ChatTemplateError.malformedMessage
                }
                return ChatMessage(role: role, content: content)
            }
            return try applyChatTemplate(
                typed,
                addGenerationPrompt: addGenerationPrompt,
                template: template
            )
        }

        /// Render a single token as text. Intended for streaming generation
        /// loops where one token is appended at a time.
        ///
        /// May return a lossy result for tokens that span a multi-byte UTF-8
        /// sequence. For correct streaming, accumulate raw bytes via
        /// `tokenBytes(_:)` and decode at codepoint boundaries.
        public func tokenToText(
            _ token: llama_token,
            renderSpecial: Bool = false
        ) -> String {
            let bytes = tokenBytes(token, renderSpecial: renderSpecial)
            return String(decoding: bytes, as: UTF8.self)
        }

        /// Render a single token as its raw UTF-8 bytes.
        func tokenBytes(
            _ token: llama_token,
            renderSpecial: Bool = false
        ) -> [UInt8] {
            var capacity: Int32 = 64
            var buffer = [CChar](repeating: 0, count: Int(capacity))

            var written = buffer.withUnsafeMutableBufferPointer { ptr in
                llama_token_to_piece(
                    vocab, token,
                    ptr.baseAddress, capacity,
                    0, renderSpecial
                )
            }

            if written < 0 {
                capacity = -written
                buffer = [CChar](repeating: 0, count: Int(capacity))
                written = buffer.withUnsafeMutableBufferPointer { ptr in
                    llama_token_to_piece(
                        vocab, token,
                        ptr.baseAddress, capacity,
                        0, renderSpecial
                    )
                }
            }

            guard written > 0 else { return [] }
            return buffer.withUnsafeBytes { raw in
                Array(raw.bindMemory(to: UInt8.self).prefix(Int(written)))
            }
        }

        private func normalize(_ token: llama_token) -> llama_token? {
            token == LLAMA_TOKEN_NULL ? nil : token
        }
    }
}
