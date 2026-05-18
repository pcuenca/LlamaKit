import LlamaSwift

/// One-shot, thread-safe initialization of the llama.cpp backend.
///
/// `llama_backend_init` must be called once per process before any other
/// llama.cpp API is used. Swift's `static let` gives us lazy, thread-safe,
/// once-only execution.
enum LlamaBackend {
    private static let initialized: Void = {
        llama_backend_init()
    }()

    static func ensureInitialized() {
        _ = initialized
    }
}
