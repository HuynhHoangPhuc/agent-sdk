//! Minimal SSE byte-stream parser.
//!
//! Implements the subset of the EventSource spec Anthropic uses:
//! - `event:` field (single line)
//! - `data:` field (multi-line accumulated with `\n` separator per spec)
//! - frames terminated by `\n\n` or `\r\n\r\n`
//! - `:` comments / blank lines ignored
//!
//! `id:` and `retry:` are accepted but discarded — Anthropic does not rely on
//! either for streaming Messages. The parser is incremental: bytes can be
//! pushed in arbitrary chunks and partial trailing data is buffered until the
//! next `\n\n`.

/// One decoded SSE frame.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SseFrame {
    /// Value of the `event:` line (empty if absent).
    pub event: String,
    /// Concatenated `data:` lines (with `\n` separators if there were
    /// multiple). Empty if absent.
    pub data: String,
}

/// Incremental parser. Push bytes; drain frames.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append more bytes to the internal buffer.
    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Drain every complete frame currently in the buffer.
    ///
    /// A "complete" frame is a span terminated by `\n\n`. Partial trailing
    /// data is retained so the next `push` can complete it.
    pub fn drain(&mut self) -> Vec<SseFrame> {
        let mut out = Vec::new();
        while let Some(boundary) = find_double_newline(&self.buf) {
            let raw = self.buf.drain(..boundary.end).collect::<Vec<u8>>();
            // Strip the boundary itself.
            let payload = &raw[..boundary.start];
            if let Some(frame) = parse_frame(payload) {
                out.push(frame);
            }
        }
        out
    }
}

/// Inclusive start / exclusive end of the `\n\n` (or `\r\n\r\n`) terminator.
struct Boundary {
    start: usize,
    end: usize,
}

fn find_double_newline(buf: &[u8]) -> Option<Boundary> {
    // Prefer the longer `\r\n\r\n` match.
    for i in 0..buf.len().saturating_sub(1) {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some(Boundary {
                start: i,
                end: i + 2,
            });
        }
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some(Boundary {
                start: i,
                end: i + 4,
            });
        }
    }
    None
}

fn parse_frame(payload: &[u8]) -> Option<SseFrame> {
    let text = std::str::from_utf8(payload).ok()?;
    let mut frame = SseFrame::default();
    let mut data_lines: Vec<&str> = Vec::new();
    for raw_line in text.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() || line.starts_with(':') {
            // Comment / blank — ignore.
            continue;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => frame.event = value.to_string(),
            "data" => data_lines.push(value),
            _ => {}
        }
    }
    if data_lines.is_empty() && frame.event.is_empty() {
        None
    } else {
        frame.data = data_lines.join("\n");
        Some(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drains_complete_frames() {
        let mut p = SseParser::new();
        p.push(b"event: message_start\ndata: {\"a\":1}\n\nevent: ping\ndata: {}\n\n");
        let frames = p.drain();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].event, "message_start");
        assert_eq!(frames[0].data, "{\"a\":1}");
        assert_eq!(frames[1].event, "ping");
    }

    #[test]
    fn buffers_partial_frame_across_pushes() {
        let mut p = SseParser::new();
        p.push(b"event: message_start\ndata: {\"a\"");
        assert!(p.drain().is_empty());
        p.push(b":1}\n\n");
        let frames = p.drain();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event, "message_start");
        assert_eq!(frames[0].data, "{\"a\":1}");
    }

    #[test]
    fn supports_crlf_terminators() {
        let mut p = SseParser::new();
        p.push(b"event: ping\r\ndata: {}\r\n\r\n");
        let frames = p.drain();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event, "ping");
        assert_eq!(frames[0].data, "{}");
    }

    #[test]
    fn ignores_comments_and_blank_lines() {
        let mut p = SseParser::new();
        p.push(b": heartbeat\nevent: ping\ndata: {}\n\n");
        let frames = p.drain();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event, "ping");
    }

    #[test]
    fn merges_multi_data_lines_with_newline() {
        let mut p = SseParser::new();
        p.push(b"data: a\ndata: b\n\n");
        let frames = p.drain();
        assert_eq!(frames[0].data, "a\nb");
    }
}
