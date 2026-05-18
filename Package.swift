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
            description: "Enables LlamaModel.from(repo:filename:) via huggingface/swift-transformers."
        ),
        .default(enabledTraits: ["Hub"]),
    ],
    dependencies: [
        .package(path: "../llama.swift"),
        .package(path: "../swift-transformers"),
    ],
    targets: [
        .target(
            name: "LlamaKit",
            dependencies: [
                .product(name: "LlamaSwift", package: "llama.swift"),
                .product(
                    name: "Hub",
                    package: "swift-transformers",
                    condition: .when(traits: ["Hub"])
                ),
            ]
        ),
        .testTarget(
            name: "LlamaKitTests",
            dependencies: [
                "LlamaKit",
                .product(
                    name: "Hub",
                    package: "swift-transformers",
                    condition: .when(traits: ["Hub"])
                ),
            ]
        ),
    ]
)
