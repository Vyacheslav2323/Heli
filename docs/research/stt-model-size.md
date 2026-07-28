# Whisper model size comparison (STT)

**Intended text:** `Hi, do you understand voice speech?`

| Model  | Transcription                              | Time (s) |
|--------|--------------------------------------------|----------|
| Large  | Hi, do you understand voice speech?        | 18.39    |
| Medium | Hi, do you understand voice pitch?         | 16.18    |
| Small  | Hi, do you understand voice pitch?         | 3.69     |
| Base   | Hi, do you understand the voice speech?    | 1.33     |
| Tiny   | Hi, do you understand the voice pitch?     | 0.75     |

## Decision

`tiny` is reasonable for now: it gets most of the phrase right, and the reasoning module will recover intent anyway.

Korean tests still need to be run before locking this in.
