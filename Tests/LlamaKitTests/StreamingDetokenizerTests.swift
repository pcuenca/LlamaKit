import Foundation
import Testing

@testable import LlamaKit

@Suite("StreamingDetokenizer")
struct StreamingDetokenizerTests {
    @Test
    func passesAsciiThrough() {
        var d = StreamingDetokenizer()
        #expect(d.consume(Array("Hello".utf8)) == "Hello")
        #expect(d.flush() == "")
    }

    @Test
    func bufferTwoByteUTF8AcrossCalls() {
        var d = StreamingDetokenizer()
        let café: [UInt8] = [0x63, 0x61, 0x66, 0xC3, 0xA9]  // "café"
        #expect(d.consume(Array(café[0..<4])) == "caf")
        #expect(d.consume([café[4]]) == "é")
    }

    @Test
    func bufferFourByteEmojiAcrossThreeCalls() {
        var d = StreamingDetokenizer()
        let smiley: [UInt8] = [0xF0, 0x9F, 0x98, 0x80]  // 😀
        #expect(d.consume([smiley[0], smiley[1]]) == "")
        #expect(d.consume([smiley[2]]) == "")
        #expect(d.consume([smiley[3]]) == "😀")
    }

    @Test
    func flushReleasesTruncatedTail() {
        var d = StreamingDetokenizer()
        let asciiThenIncomplete: [UInt8] = [0x68, 0x69, 0xF0, 0x9F]  // "hi" + leading half of 😀
        #expect(d.consume(asciiThenIncomplete) == "hi")
        let tail = d.flush()
        #expect(!tail.isEmpty)
        #expect(tail.unicodeScalars.contains(Unicode.Scalar(0xFFFD)!))
    }

    @Test
    func emptyConsumeIsEmpty() {
        var d = StreamingDetokenizer()
        #expect(d.consume([]) == "")
        #expect(d.flush() == "")
    }
}
