#if Hub
    import Foundation
    import Hub

    extension LlamaModel {
        public enum HubError: Error, CustomStringConvertible {
            case noMatchingFile(repo: String, pattern: String)

            public var description: String {
                switch self {
                case .noMatchingFile(let repo, let pattern):
                    return "No file matching \(pattern) in \(repo)"
                }
            }
        }

        /// Downloads a GGUF file from the Hugging Face Hub and loads it.
        ///
        /// The first matching `.gguf` file inside the snapshot is used. The
        /// snapshot is cached under `HubApi`'s `downloadBase` and reused on
        /// subsequent calls.
        ///
        /// - Parameters:
        ///   - repo: A Hub repo id, e.g. `"unsloth/SmolLM2-135M-Instruct-GGUF"`.
        ///   - filename: A glob pattern matching the target GGUF file.
        ///   - revision: Branch, tag, or commit. Defaults to `"main"`.
        ///   - hubApi: The `HubApi` to use. Defaults to a shared instance.
        ///   - parameters: Model load parameters.
        ///   - progress: Optional progress callback.
        public static func from(
            repo: String,
            filename: String,
            revision: String = "main",
            hubApi: HubApi = HubApi(),
            parameters: Parameters = .default,
            progress: @escaping @Sendable (Progress) -> Void = { _ in }
        ) async throws -> LlamaModel {
            let snapshotURL = try await hubApi.snapshot(
                from: repo,
                revision: revision,
                matching: filename,
                progressHandler: progress
            )

            let fileURL = try locateGGUF(in: snapshotURL, pattern: filename, repo: repo)
            return try LlamaModel(contentsOf: fileURL, parameters: parameters)
        }

        private static func locateGGUF(
            in snapshot: URL,
            pattern: String,
            repo: String
        ) throws -> URL {
            let enumerator = FileManager.default.enumerator(
                at: snapshot,
                includingPropertiesForKeys: [.isRegularFileKey]
            )
            while let url = enumerator?.nextObject() as? URL {
                if url.pathExtension.lowercased() == "gguf" {
                    return url
                }
            }
            throw HubError.noMatchingFile(repo: repo, pattern: pattern)
        }
    }
#endif
