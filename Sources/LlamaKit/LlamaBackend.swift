import LlamaSwift

/// One-shot, thread-safe initialization of the llama.cpp backend.
enum LlamaBackend {
    private static let initialized: Void = {
        llama_backend_init()
    }()

    static func ensureInitialized() {
        _ = initialized
    }
}
