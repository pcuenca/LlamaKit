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
        @Test
        func loadsSmolLM2FromHub() async throws {
            let model = try await LlamaModel.from(
                repo: "unsloth/SmolLM2-135M-Instruct-GGUF",
                filename: "*Q2_K.gguf"
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
