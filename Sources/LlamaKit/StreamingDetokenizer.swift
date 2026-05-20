import Foundation

/// Buffers raw UTF-8 bytes across tokens so callers see clean codepoints.
///
/// llama.cpp's `llama_token_to_piece` returns the raw bytes that a single
/// token contributes to the rendered text. For BPE / byte-fallback tokenizers,
/// a multi-byte UTF-8 codepoint (CJK, emoji, accented characters) can be split
/// across two or more tokens. Decoding each token's bytes in isolation would
/// emit `U+FFFD` for the incomplete halves. `StreamingDetokenizer` accumulates
/// bytes and only releases the prefix that ends on a complete codepoint
/// boundary; the trailing partial bytes carry over to the next `consume`.
struct StreamingDetokenizer {
    private var pending: [UInt8] = []

    mutating func consume(_ bytes: [UInt8]) -> String {
        pending.append(contentsOf: bytes)
        let boundary = completeByteCount()
        guard boundary > 0 else { return "" }
        let chunk = pending.prefix(boundary)
        pending.removeFirst(boundary)
        return String(decoding: chunk, as: UTF8.self)
    }

    /// Emit whatever's left, allowing the standard library to substitute
    /// `U+FFFD` for any genuinely truncated codepoint at the end of the stream.
    mutating func flush() -> String {
        defer { pending.removeAll(keepingCapacity: false) }
        return String(decoding: pending, as: UTF8.self)
    }

    /// Returns the largest prefix length of `pending` that ends on a complete
    /// UTF-8 codepoint. Walks backwards to find the last leading byte and
    /// checks whether all of its continuation bytes have arrived.
    private func completeByteCount() -> Int {
        guard !pending.isEmpty else { return 0 }

        var i = pending.count
        while i > 0 {
            i -= 1
            let byte = pending[i]

            if byte & 0x80 == 0 {
                return i + 1
            }
            if byte & 0xC0 == 0x80 {
                continue
            }

            let expected: Int
            if byte & 0xE0 == 0xC0 { expected = 2 }
            else if byte & 0xF0 == 0xE0 { expected = 3 }
            else if byte & 0xF8 == 0xF0 { expected = 4 }
            else { return i + 1 }

            let actual = pending.count - i
            return actual >= expected ? pending.count : i
        }

        return 0
    }
}
