import Foundation

/// A single message in a chat conversation, as consumed by chat templates.
public struct ChatMessage: Sendable, Hashable {
    /// The message's role. Common values: `"system"`, `"user"`,
    /// `"assistant"`, `"tool"`. The set of accepted roles is template-specific.
    public var role: String

    /// The message's textual content.
    public var content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

extension ChatMessage {
    public static func system(_ content: String) -> ChatMessage {
        ChatMessage(role: "system", content: content)
    }

    public static func user(_ content: String) -> ChatMessage {
        ChatMessage(role: "user", content: content)
    }

    public static func assistant(_ content: String) -> ChatMessage {
        ChatMessage(role: "assistant", content: content)
    }

    public static func tool(_ content: String) -> ChatMessage {
        ChatMessage(role: "tool", content: content)
    }
}
