# whisper-distill-sr

Distils Whisper large-v3-turbo into Whisper small for Serbian speech, on one Apple Silicon Mac.
The student keeps the size and speed of `small` (488 MB in whisper.cpp format).

**Download the model:** [rawmind/whisper-small-sr-distill](https://huggingface.co/rawmind/whisper-small-sr-distill) on Hugging Face.

## Serbian tests

| Serbian test | Clips | Student | Whisper small | large-v3-turbo |
|---|---|---|---|---|
| FLEURS, clean read speech | 400 | **24.6** (23.2–25.9) | 38.1 (36.6–39.7) | 13.2 (12.4–14.2) |
| Parliament, real speech | 150 | **22.1** (20.2–24.0) | 28.5 (26.2–30.7) | 10.1 (8.8–11.5) |
| FLEURS + pink noise, 10 dB | 100 | **29.1** (26.6–31.8) | 43.0 (40.2–45.9) | 16.9 (14.9–19.0) |
| FLEURS + pink noise, 5 dB | 100 | **36.3** (33.2–39.4) | 50.5 (47.5–53.7) | 19.9 (17.7–22.2) |
| FLEURS, shortened context | 100 | **27.9** (24.4–32.2) | 39.3 (36.3–42.5) | 22.1 (16.7–28.2) |
| Parliament, shortened context | 100 | **22.4** (20.0–25.0) | 27.4 (24.7–30.2) | 15.9 (12.9–19.7) |

Word error rate in %, with a 95% bootstrap range over clips. Same size as `small` and the same speed: 212 ms against 211 ms per clip on an M4 Max GPU. The student beats `small` in all six tests; in the last one the ranges touch. Turbo is far ahead but 1.6 GB and about 2.3 times slower.

- **Parliament:** 150 clips, 28 speakers, from one file of [ParlaSpeech-RS](https://huggingface.co/datasets/classla/ParlaSpeech-RS) (CC-BY-4.0). None were in the student's training data, which was FLEURS only. Its transcripts are official records, not always word for word, so compare models, not totals. Whisper's own training data isn't public, so turbo's 10.1 could benefit from overlap.
- **Shortened context:** whisper.cpp's `audio_ctx` set to the clip length plus one second, as Titl does on slow machines.
- **Silence:** on five clips with no speech (digital silence and plain noise) all three models wrote text. The student wrote `(Šeštaj)` on four and a line of mixed-script gibberish on the silent one, `small` wrote `[Muzika]` or `*svijedna*`, and turbo wrote `Hvala.`. Filter out silent audio before using any of them.

## Three languages

Word error rate in %, lower is better. 100 FLEURS test clips per language, whisper.cpp 1.9.6, language given, Serbian compared in Latin script.

| Model | Serbian | Croatian | English |
|---|---|---|---|
| Whisper small | 38.2 | 35.2 | 7.0 |
| **Student** | **24.6** | **23.6** | 7.5 |
| Parakeet TDT 0.6B v3, q8_0 | 21.2 | 10.9 | 5.8 |
| Whisper large-v3-turbo (teacher) | 12.4 | 11.1 | 5.5 |

Raw numbers are in `results/`: `serbian_tests.txt` for the Serbian tests, `student.tsv` and `baselines.tsv` for this table. Parakeet's Serbian output uses Croatian forms, which this score counts as errors.

## Limits

- Trained on FLEURS only, clean read speech. Noise is artificial pink noise; music, video and several people talking are untested.
- Trained on Serbian (2,944 clips), Croatian (200) and English (200). Other languages are untested and may be worse than `small`.
- Language detection is poor: on 50 Serbian clips it answered Croatian 35 times and Serbian 14. Set the language.
- In the three-language table each test is 100 clips, so a difference of about 1 point means nothing.
- It writes text on audio with no speech, as the other models do.

## Method

1. The teacher reads the human transcript of each clip and its 64 most likely next tokens are stored once.
2. The student learns from the transcript (cross-entropy) and from those tokens (KL, temperature 2), weighted equally, with Serbian in Latin script.
3. AdamW, learning rate 2e-5 with cosine decay, batch 4 as two micro-batches, 1,353 steps (1.6 epochs) in 35 minutes on an M4 Max.

Training stops early if disk space runs low, or if the Mac is on battery below 20%. Sentences that appear in the test sets are excluded; none did.

## Run it

Needs macOS with Apple Silicon, and `whisper-server` from whisper.cpp for the benchmark.

```sh
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python prepare.py
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 python distill.py
./convert.sh
python benchmark.py student=models/ggml-small-sr-distill.bin --checks
python serbian_tests.py student=models/ggml-small-sr-distill.bin
```

`serbian_tests.py` downloads the parliament file and the 400 FLEURS clips, about 800 MB, on its first run.
`prepare.py` downloads about 3 GB in total and keeps 0.8 GB. The teacher pass takes about 24 minutes and is cached in `work/teacher.pt`.
The watermark variables cap GPU memory at about 18 GB on a 36 GB Mac. Without a cap, an early run swapped until the disk filled up.

Use the model with whisper.cpp:

```sh
whisper-cli -m models/ggml-small-sr-distill.bin -l sr -f audio.wav
```

## License

Code: MIT. Weights derive from [whisper-small](https://huggingface.co/openai/whisper-small) (Apache-2.0) and [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT). Training data: [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0).
