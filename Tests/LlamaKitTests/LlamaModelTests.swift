import Foundation
import Testing

@testable import LlamaKit

@Suite("LlamaModel")
struct LlamaModelTests {
    @Test
    func loadFailsForMissingFile() {
        let url = URL(fileURLWithPath: "/tmp/does-not-exist.gguf")
        #expect(throws: LlamaModel.LoadError.self) {
            _ = try LlamaModel(contentsOf: url)
        }
    }

    #if Hub
        /// Downloads a tiny GGUF and verifies the model's reported metadata.
        ///
        /// Requires network and the `Hub` trait. Skipped in CI unless
        /// `LLAMAKIT_NETWORK_TESTS=1` is set, since the first run pulls
        /// ~80 MB.
        @Test(
            .enabled(
                if: ProcessInfo.processInfo.environment["LLAMAKIT_NETWORK_TESTS"] == "1",
                "set LLAMAKIT_NETWORK_TESTS=1 to enable Hub-backed tests"
            )
        )
        func loadsSmolLM2FromHub() async throws {
            let model = try await LlamaModel.from(
                repo: "unsloth/SmolLM2-135M-Instruct-GGUF",
                filename: "*Q4_K_M.gguf"
            )

            #expect(model.parameterCount > 0)
            #expect(model.vocabularySize > 0)
            #expect(model.trainingContextLength > 0)
            #expect(model.embeddingDimension > 0)
            #expect(model.layerCount > 0)
            #expect(model.hasDecoder)
            #expect(model.chatTemplate != nil)
        }
    #endif
}
