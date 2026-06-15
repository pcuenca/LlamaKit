#if Hub
    import Foundation
    import HuggingFace

    extension LlamaModel {
        public enum HubError: Error, CustomStringConvertible {
            case invalidRepo(String)
            case noMatchingFile(repo: String, pattern: String)
            case repoNotFound(repo: String)

            public var description: String {
                switch self {
                case .invalidRepo(let repo):
                    return "Invalid Hub repo identifier '\(repo)' — expected 'namespace/name'"
                case .noMatchingFile(let repo, let pattern):
                    return "No file matching \(pattern) in \(repo)"
                case .repoNotFound(let repo):
                    return "Hugging Face repo '\(repo)' not found"
                }
            }
        }

        /// Downloads a GGUF file from the Hugging Face Hub and loads it.
        ///
        /// The first matching `.gguf` file inside the snapshot is used. The
        /// snapshot is cached on disk and reused on subsequent calls.
        ///
        /// - Parameters:
        ///   - repo: A Hub repo id, e.g. `"unsloth/SmolLM2-135M-Instruct-GGUF"`.
        ///   - filename: A glob pattern matching the target GGUF file.
        ///   - revision: Branch, tag, or commit. Defaults to `"main"`.
        ///   - hubClient: The `HubClient` to use. Defaults to the shared instance.
        ///   - parameters: Model load parameters.
        ///   - progress: Optional progress callback.
        public static func from(
            repo: String,
            filename: String,
            revision: String = "main",
            hubClient: HubClient = .default,
            parameters: Parameters = .default,
            progress: (@MainActor @Sendable (Progress) -> Void)? = nil
        ) async throws -> LlamaModel {
            guard let repoID = Repo.ID(rawValue: repo) else {
                throw HubError.invalidRepo(repo)
            }
            let snapshotURL: URL
            do {
                snapshotURL = try await hubClient.downloadSnapshot(
                    of: repoID,
                    revision: revision,
                    matching: [filename],
                    progressHandler: progress
                )
            } catch let httpError as HTTPClientError {
                if case let .responseError(response, _) = httpError,
                    response.statusCode == 404
                {
                    throw HubError.repoNotFound(repo: repo)
                }
                throw httpError
            }
            let fileURL = try locateGGUF(in: snapshotURL, pattern: filename, repo: repo)
            return try LlamaModel(contentsOf: fileURL, parameters: parameters)
        }

        private static func locateGGUF(
            in snapshot: URL,
            pattern: String,
            repo: String
        ) throws -> URL {
            let glob = NSPredicate(format: "self LIKE[c] %@", pattern)
            let enumerator = FileManager.default.enumerator(
                at: snapshot,
                includingPropertiesForKeys: [.isRegularFileKey]
            )
            while let url = enumerator?.nextObject() as? URL {
                guard url.pathExtension.lowercased() == "gguf" else { continue }
                guard glob.evaluate(with: url.lastPathComponent) else { continue }
                return url
            }
            throw HubError.noMatchingFile(repo: repo, pattern: pattern)
        }
    }
#endif
