import LlamaSwift

/// A recipe describing how to sample the next token from a sequence of logits.
///
/// `Sampler` is a value type: an immutable, `Sendable` description of an
/// ordered list of sampling stages. Each generation builds a fresh native
/// sampler from the recipe and releases it when done — the recipe itself
/// can be reused across many generations without leaking state.
///
/// Order matters: llama.cpp evaluates stages left-to-right, so put filters
/// (top-k, top-p) before temperature, and finish with a distribution stage
/// (`.distribution`, `.greedy`, or `.mirostat`) that actually picks the token.
public struct Sampler: Sendable {
    public enum Stage: Sendable, Hashable {
        /// Argmax — pick the token with the highest logit.
        case greedy

        /// Sample from the current distribution using the given seed.
        case distribution(seed: UInt32)

        /// Scale logits by `1 / value`. Higher values flatten the
        /// distribution; lower values sharpen it.
        case temperature(Float)

        /// Keep only the `k` highest-logit tokens.
        case topK(Int32)

        /// Keep the smallest set of tokens whose cumulative probability
        /// exceeds `p`. `minKeep` guarantees at least N tokens survive.
        case topP(Float, minKeep: Int32)

        /// Drop tokens whose probability is below `p × max(probabilities)`.
        case minP(Float, minKeep: Int32)

        /// Locally typical sampling (Meister et al. 2022).
        case typical(Float, minKeep: Int32)

        /// Apply repetition, frequency, and presence penalties to recently
        /// generated tokens. `lastN = 0` disables; `lastN = -1` uses the full
        /// context.
        case penalties(
            lastN: Int32,
            repeatPenalty: Float,
            frequencyPenalty: Float,
            presencePenalty: Float
        )

        /// Mirostat 2.0 adaptive sampling (Basu et al. 2007.14966). Targets a
        /// constant perplexity `tau`; `eta` is the learning rate.
        case mirostat(tau: Float, eta: Float, seed: UInt32)
    }

    public let stages: [Stage]

    public init(stages: [Stage]) {
        self.stages = stages
    }
}

// MARK: - Presets

extension Sampler {
    /// Deterministic argmax sampling. The same prompt always produces the
    /// same continuation.
    public static let greedy = Sampler(stages: [.greedy])

    /// Standard temperature sampling with optional top-k and top-p filters.
    ///
    /// Defaults match llama.cpp's commonly used preset: temperature 0.8,
    /// top-k 40, top-p 0.95, and a random seed. Pass `nil` for `topK` or
    /// `topP` to disable that filter.
    public static func temperature(
        _ value: Float = 0.8,
        topK: Int32? = 40,
        topP: Float? = 0.95,
        seed: UInt32 = .random(in: .min ... .max)
    ) -> Sampler {
        var stages: [Stage] = []
        if let topK { stages.append(.topK(topK)) }
        if let topP { stages.append(.topP(topP, minKeep: 1)) }
        stages.append(.temperature(value))
        stages.append(.distribution(seed: seed))
        return Sampler(stages: stages)
    }

    /// Min-p sampling — keeps tokens whose probability is at least `p` times
    /// the most likely token. Often produces more coherent output than top-p
    /// at low temperatures.
    public static func minP(
        _ p: Float = 0.05,
        temperature: Float = 0.8,
        seed: UInt32 = .random(in: .min ... .max)
    ) -> Sampler {
        Sampler(stages: [
            .minP(p, minKeep: 1),
            .temperature(temperature),
            .distribution(seed: seed),
        ])
    }
}

// MARK: - Native handle

extension Sampler {
    /// An ARC-owned native handle built from a `Sampler` recipe. Internal to
    /// the package — generation code holds an `Instance` for the lifetime of
    /// one run; the deinit releases the native chain.
    final class Instance {
        let pointer: UnsafeMutablePointer<llama_sampler>

        init(_ sampler: Sampler) {
            let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!
            for stage in sampler.stages {
                llama_sampler_chain_add(chain, stage.makeC())
            }
            self.pointer = chain
        }

        deinit {
            llama_sampler_free(pointer)
        }
    }

    func instantiate() -> Instance {
        Instance(self)
    }
}

extension Sampler.Stage {
    fileprivate func makeC() -> UnsafeMutablePointer<llama_sampler> {
        switch self {
        case .greedy:
            return llama_sampler_init_greedy()
        case .distribution(let seed):
            return llama_sampler_init_dist(seed)
        case .temperature(let t):
            return llama_sampler_init_temp(t)
        case .topK(let k):
            return llama_sampler_init_top_k(k)
        case .topP(let p, let minKeep):
            return llama_sampler_init_top_p(p, Int(minKeep))
        case .minP(let p, let minKeep):
            return llama_sampler_init_min_p(p, Int(minKeep))
        case .typical(let p, let minKeep):
            return llama_sampler_init_typical(p, Int(minKeep))
        case .penalties(let lastN, let repeatPenalty, let frequencyPenalty, let presencePenalty):
            return llama_sampler_init_penalties(
                lastN, repeatPenalty, frequencyPenalty, presencePenalty
            )
        case .mirostat(let tau, let eta, let seed):
            return llama_sampler_init_mirostat_v2(seed, tau, eta)
        }
    }
}
