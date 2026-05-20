// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "LlamaKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .watchOS(.v9),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "LlamaKit",
            targets: ["LlamaKit"]
        )
    ],
    traits: [
        .trait(
            name: "Hub",
            description: "Enables LlamaModel.from(repo:filename:) via huggingface/swift-huggingface."
        ),
        .default(enabledTraits: ["Hub"]),
    ],
    dependencies: [
        .package(url: "https://github.com/mattt/llama.swift", .upToNextMajor(from: "2.9190.0")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.8.1"),
    ],
    targets: [
        .target(
            name: "LlamaKit",
            dependencies: [
                .product(name: "LlamaSwift", package: "llama.swift"),
                .product(
                    name: "HuggingFace",
                    package: "swift-huggingface",
                    condition: .when(traits: ["Hub"])
                ),
            ]
        ),
        .testTarget(
            name: "LlamaKitTests",
            dependencies: [
                "LlamaKit",
                .product(
                    name: "HuggingFace",
                    package: "swift-huggingface",
                    condition: .when(traits: ["Hub"])
                ),
            ]
        ),
    ]
)
