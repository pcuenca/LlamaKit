import Foundation
import LlamaSwift

/// One-shot, thread-safe initialization of the llama.cpp backend.
public enum LlamaBackend {
    /// Whether to forward llama.cpp / ggml log messages to stderr.
    ///
    /// Set to `false` before any other LlamaKit API call to also silence backend startup messages.
    nonisolated(unsafe) public static var loggingEnabled: Bool = true

    private static let logCallback: ggml_log_callback = { _, text, _ in
        guard LlamaBackend.loggingEnabled, let text else { return }
        fputs(String(cString: text), stderr)
    }

    private static let initialized: Void = {
        llama_log_set(logCallback, nil)
        llama_backend_init()
    }()

    static func ensureInitialized() {
        _ = initialized
    }
}
