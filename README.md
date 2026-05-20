# LlamaKit

Swift wrapper around [llama.cpp](https://github.com/ggml-org/llama.cpp), built on the [llama.swift](https://github.com/mattt/llama.swift) XCFramework, and with Hub integration via [swift-huggingface](https://github.com/huggingface/swift-huggingface).

Provides idiomatic Swift types for loading GGUF models, tokenizing, applying
chat templates, sampling, and streaming generation.

## Load a model from the Hugging Face Hub

```swift
import LlamaKit

let model = try await LlamaModel.from(
    repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename: "*Q8_0.gguf"
)
```

The first matching `.gguf` file in the snapshot is loaded.

To load a local file:

```swift
let model = try LlamaModel(contentsOf: URL(fileURLWithPath: "model.gguf"))
```

## Apply a chat template

```swift
let prompt = try model.tokenizer.applyChatTemplate([
    .system("You are a helpful assistant."),
    .user("Explain entanglement in one sentence."),
])
```

Renders the conversation using the chat template embedded in the GGUF.
Pass `addGenerationPrompt: false` if you don't want the assistant-turn marker appended.

## Generate text

```swift
let session = try LlamaSession(model: model)

let response = try await session.complete(
    prompt: prompt,
    sampler: .temperature(0.7),
    maxTokens: 256
)
```

Or stream tokens as they're sampled:

```swift
for try await chunk in session.generate(prompt: prompt) {
    print(chunk, terminator: "")
}
```

Use `Sampler.greedy` for deterministic output.
`Sampler.minP(0.05)` and custom `Sampler(stages: [...])` chains are also
supported.

## Example: `kitchat`

`Examples/Chat/` contains a [small interactive chat CLI](https://github.com/pcuenca/LlamaKit/blob/main/Examples/Chat/Sources/kitchat/ChatCommand.swift) built on LlamaKit:

```bash
swift run kitchat --hf Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

## Installation

```swift
.package(url: "https://github.com/pcuenca/LlamaKit", from: "0.1.0")
```

## Requirements

- Swift 6.1+
- macOS 13+ / iOS 16+ / visionOS 1+
