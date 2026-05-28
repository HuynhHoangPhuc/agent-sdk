//! Minimal SSE byte-stream parser shared by both OpenAI APIs.
//!
//! Implements the subset of the EventSource spec OpenAI uses:
//! - `event:` field (single line, present on the Responses API; absent on Chat)
//! - `data:` field (multi-line accumulated with `\n` separator per spec)
//! - frames terminated by `\n\n` or `\r\n\r\n`
//! - `:` comments / blank lines ignored
//!
//! `id:` and `retry:` are accepted but discarded.
//!
//! Identical in shape to the Anthropic provider's parser; duplicated rather
//! than extracted to a shared crate so each provider can evolve independently.

/// One decoded SSE frame.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SseFrame {
    /// Value of the `event:` line (empty if absent).
    pub event: String,
    /// Concatenated `data:` lines.
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

    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    pub fn drain(&mut self) -> Vec<SseFrame> {
        let mut out = Vec::new();
        while let Some(boundary) = find_double_newline(&self.buf) {
            let raw = self.buf.drain(..boundary.end).collect::<Vec<u8>>();
            let payload = &raw[..boundary.start];
            if let Some(frame) = parse_frame(payload) {
                out.push(frame);
            }
        }
        out
    }
}

struct Boundary {
    start: usize,
    end: usize,
}

fn find_double_newline(buf: &[u8]) -> Option<Boundary> {
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
        p.push(b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\n");
        let frames = p.drain();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].data, "{\"a\":1}");
        assert_eq!(frames[1].data, "{\"b\":2}");
    }

    #[test]
    fn buffers_partial_frame_across_pushes() {
        let mut p = SseParser::new();
        p.push(b"data: {\"a\"");
        assert!(p.drain().is_empty());
        p.push(b":1}\n\n");
        let frames = p.drain();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, "{\"a\":1}");
    }

    #[test]
    fn keeps_event_line_for_typed_apis() {
        let mut p = SseParser::new();
        p.push(b"event: response.output_text.delta\ndata: {\"delta\":\"hi\"}\n\n");
        let frames = p.drain();
        assert_eq!(frames[0].event, "response.output_text.delta");
        assert_eq!(frames[0].data, "{\"delta\":\"hi\"}");
    }
}
