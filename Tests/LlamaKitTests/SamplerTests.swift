import Foundation
import LlamaSwift
import Testing

@testable import LlamaKit

@Suite("Sampler")
struct SamplerTests {
    @Test
    func greedyHasOneStage() {
        let sampler = Sampler.greedy
        #expect(sampler.stages == [.greedy])
    }

    @Test
    func temperaturePresetBuildsExpectedChain() {
        let sampler = Sampler.temperature(0.7, topK: 50, topP: 0.9, seed: 42)
        #expect(sampler.stages.count == 4)
        #expect(sampler.stages[0] == .topK(50))
        #expect(sampler.stages[1] == .topP(0.9, minKeep: 1))
        #expect(sampler.stages[2] == .temperature(0.7))
        #expect(sampler.stages[3] == .distribution(seed: 42))
    }

    @Test
    func temperaturePresetSkipsDisabledFilters() {
        let sampler = Sampler.temperature(1.0, topK: nil, topP: nil, seed: 0)
        #expect(sampler.stages == [.temperature(1.0), .distribution(seed: 0)])
    }

    @Test
    func instantiateMatchesStageCount() {
        let sampler = Sampler(stages: [
            .penalties(lastN: 64, repeatPenalty: 1.1, frequencyPenalty: 0.0, presencePenalty: 0.0),
            .topK(40),
            .topP(0.95, minKeep: 1),
            .temperature(0.8),
            .distribution(seed: 7),
        ])

        let instance = sampler.instantiate()
        #expect(llama_sampler_chain_n(instance.pointer) == Int32(sampler.stages.count))
    }

    @Test
    func mirostatStageInstantiates() {
        let sampler = Sampler(stages: [
            .temperature(1.0),
            .mirostat(tau: 5.0, eta: 0.1, seed: 1),
        ])

        let instance = sampler.instantiate()
        #expect(llama_sampler_chain_n(instance.pointer) == 2)
    }
}
