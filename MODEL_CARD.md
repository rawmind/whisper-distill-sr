---
license: mit
language:
- sr
- hr
- en
base_model: openai/whisper-small
pipeline_tag: automatic-speech-recognition
library_name: transformers
datasets:
- google/fleurs
tags:
- whisper
- distillation
- serbian
- whisper.cpp
---

# whisper-small-sr-distill

Whisper small, distilled from Whisper large-v3-turbo for Serbian. Same size and speed as `small`, with fewer errors in Serbian and Croatian.
An experiment, not a production model.

**Training code and raw results:** [rawmind/whisper-distill-sr](https://github.com/rawmind/whisper-distill-sr) on GitHub.

## Serbian tests

| Serbian test | Clips | Student | Whisper small | large-v3-turbo |
|---|---|---|---|---|
| FLEURS, clean read speech | 400 | **24.6** (23.2–25.9) | 38.1 (36.6–39.7) | 13.2 (12.4–14.2) |
| Parliament, real speech | 150 | **22.1** (20.2–24.0) | 28.5 (26.2–30.7) | 10.1 (8.8–11.5) |
| FLEURS + pink noise, 10 dB | 100 | **29.1** (26.6–31.8) | 43.0 (40.2–45.9) | 16.9 (14.9–19.0) |
| FLEURS + pink noise, 5 dB | 100 | **36.3** (33.2–39.4) | 50.5 (47.5–53.7) | 19.9 (17.7–22.2) |
| FLEURS, shortened context | 100 | **27.9** (24.4–32.2) | 39.3 (36.3–42.5) | 22.1 (16.7–28.2) |
| Parliament, shortened context | 100 | **22.4** (20.0–25.0) | 27.4 (24.7–30.2) | 15.9 (12.9–19.7) |

Word error rate in % (lower is better), whisper.cpp 1.9.6, with a 95% bootstrap range over clips. Same size and speed as `small`: 212 ms against 211 ms per clip on an M4 Max GPU. The student beats `small` in all six tests; in the last one the ranges overlap slightly. Turbo is far ahead but 1.6 GB and about 2.3 times slower.

- **Parliament:** 150 clips, 28 speakers, from one file of [ParlaSpeech-RS](https://huggingface.co/datasets/classla/ParlaSpeech-RS) (CC-BY-4.0). None were in the student's training data, which was FLEURS only. Its transcripts are official records, not always word for word, so compare models, not totals. Whisper's own training data isn't public, so turbo's 10.1 could benefit from overlap.
- **Shortened context:** whisper.cpp's `audio_ctx` set to the clip length plus one second, as Titl does on slow machines.
- **Silence:** on five clips with no speech (digital silence and plain noise) all three models wrote text. The student wrote `(Šeštaj)` or `(„eštaj)` on four and a line of mixed-script gibberish on the silent one, `small` wrote `[Muzika]` or `*svijedna*`, and turbo wrote `Hvala.` or `Hvala što pratite kanal.`. Filter out silent audio before using any of them.

## Three languages

Word error rate in %, 100 FLEURS test clips per language, whisper.cpp 1.9.6, language given, Serbian compared in Latin script:

| Model | Serbian | Croatian | English |
|---|---|---|---|
| Whisper small | 38.2 | 35.2 | 7.0 |
| **This model** | **24.6** | **23.6** | 7.5 |
| Whisper large-v3-turbo (teacher) | 12.4 | 11.1 | 5.5 |

## Use

Files: the Transformers weights (fp16), and `ggml-small-sr-distill.bin` for whisper.cpp.

```sh
whisper-cli -m ggml-small-sr-distill.bin -l sr -f audio.wav
```

Set the language. Serbian training transcripts were converted to Latin script, so expect mostly Latin output.

## Limits

- Trained on FLEURS only, clean read speech. Noise is artificial pink noise; music, video and several people talking are untested.
- Trained on 2,944 Serbian, 200 Croatian and 200 English clips. Other languages are untested and may be worse than `small`.
- Language detection is poor: on 50 Serbian clips it answered Croatian 35 times and Serbian 14.
- In the three-language table each test is 100 clips, so a difference of about 1 point means nothing.
- It writes text on audio with no speech, as the other models do.

## Training

Cross-entropy on the FLEURS human transcripts and KL divergence to the teacher's 64 most likely tokens (temperature 2), weighted equally. AdamW at 2e-5 with cosine decay, 1,353 steps of batch 4 (1.6 epochs) on an Apple M4 Max.

Derived from [whisper-small](https://huggingface.co/openai/whisper-small) (Apache-2.0), with [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT) as the teacher. Trained on [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0).
