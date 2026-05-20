// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "chat",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../.."),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .executableTarget(
            name: "kitchat",
            dependencies: [
                .product(name: "LlamaKit", package: "LlamaKit"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        )
    ]
)
