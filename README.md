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

Word error rate in %, with a 95% bootstrap range over clips. Same size as `small` and the same speed: 212 ms against 211 ms per clip on an M4 Max GPU. The student beats `small` in all six tests; in the last one the ranges overlap slightly. Turbo is far ahead but 1.6 GB and about 2.3 times slower.

- **Parliament:** 150 clips, 28 speakers, from one file of [ParlaSpeech-RS](https://huggingface.co/datasets/classla/ParlaSpeech-RS) (CC-BY-4.0). None were in the student's training data, which was FLEURS only. Its transcripts are official records, not always word for word, so compare models, not totals. Whisper's own training data isn't public, so turbo's 10.1 could benefit from overlap.
- **Shortened context:** whisper.cpp's `audio_ctx` set to the clip length plus one second, as Titl does on slow machines.
- **Silence:** on five clips with no speech (digital silence and plain noise) all three models wrote text. The student wrote `(Šeštaj)` or `(„eštaj)` on four and a line of mixed-script gibberish on the silent one, `small` wrote `[Muzika]` or `*svijedna*`, and turbo wrote `Hvala.` or `Hvala što pratite kanal.`. Filter out silent audio before using any of them.

## Croatian tests

| Croatian test | Clips | Student | Whisper small | large-v3-turbo |
|---|---|---|---|---|
| FLEURS, clean read speech | 400 | **24.3** (23.0–25.7) | 35.5 (34.0–37.1) | 11.4 (10.4–12.4) |
| Parliament, real speech | 150 | **29.6** (27.1–31.8) | 33.1 (30.4–35.6) | 13.6 (11.9–15.4) |
| FLEURS + pink noise, 10 dB | 100 | **37.9** (35.3–40.7) | 51.8 (48.9–54.8) | 17.7 (15.4–19.9) |
| FLEURS + pink noise, 5 dB | 100 | **52.6** (49.2–56.1) | 68.3 (65.0–71.6) | 29.9 (26.9–33.1) |
| FLEURS, shortened context | 100 | **25.8** (23.0–28.9) | 35.8 (32.8–39.0) | 16.7 (12.5–21.3) |
| Parliament, shortened context | 100 | **30.0** (27.2–32.9) | 33.5 (30.4–36.8) | 17.4 (14.3–20.7) |

Same units and method as the Serbian tests. The student beats `small` in all six, but by less than in Serbian on real speech. On the 150 parliament clips its range overlaps `small`'s, so the two were compared clip by clip: the student is better by 3.5 points (95% 2.0–5.0), with fewer errors on 86 clips and more on 39. Those clips are from one file of [ParlaSpeech-HR](https://huggingface.co/datasets/classla/ParlaSpeech-HR), 44 speakers, and none of the FLEURS test sentences are in the training data.

**Serbian forms:** the student sometimes writes a Serbian form, such as *predsednik* for *predsjednik*. On parliament that was 18 of 454 words containing *je*, against 11 for `small` and 2 for turbo. On FLEURS it was 18 of 1,189, against 21 and 8. This counts misspellings such as *dule* for *dulje* too.

## Quantized q8_0

`ggml-small-sr-distill-q8_0.bin` is 264 MB instead of 488 MB and scores the same as f16 in these tests (word error rate in %, same clips as above):

| Test | Clips per language | Serbian q8_0 | Serbian f16 | Croatian q8_0 | Croatian f16 |
|---|---|---|---|---|---|
| FLEURS | 400 | 24.4 | 24.6 | 24.3 | 24.3 |
| Parliament | 150 | 22.2 | 22.1 | 29.6 | 29.6 |
| FLEURS + pink noise, 10 dB | 100 | 29.4 | 29.1 | 37.8 | 37.9 |
| FLEURS + pink noise, 5 dB | 100 | 36.4 | 36.3 | 52.1 | 52.6 |
| FLEURS, shortened context | 100 | 28.5 | 27.9 | 26.4 | 25.8 |
| Parliament, shortened context | 100 | 22.1 | 22.4 | 30.2 | 30.0 |

On an M4 Max CPU with 4 threads and no Accelerate it was 2.6 times as fast as f16, and used 443–497 MB of memory against 653 MB. Speed on Intel and AMD is untested. These scores are from the Metal build, like the f16 ones.

## CPU

Serbian, 30 FLEURS and 30 parliament clips (700 s), whisper.cpp 1.9.6 built without Metal, Accelerate or BLAS, 4 threads, on an M4 Max. Speed is audio time divided by processing time, so 4.9 means 4.9 times faster than real time. Shortened context is `audio_ctx` set to the clip length plus one second.

| Model | MB | Speed, full | Speed, shortened | Typical clip, full | Memory, full | WER, full: FLEURS / parliament |
|---|---|---|---|---|---|---|
| Parakeet TDT 0.6B v3, q8_0 | 669 | 37 | - | 0.3 s | - | 21.0 / 13.7 |
| **Student, q8_0** | 264 | **12.8** | **28.6** | 0.9 s | 443 MB | 28.7 / 21.5 |
| Whisper small, q8_0 | 264 | 12.7 | 28.6 | 0.9 s | 443 MB | 41.6 / 28.2 |
| Student, q5_1 | 190 | 5.8 | 13.3 | 2.0 s | 370 MB | 28.7 / 22.2 |
| Whisper small, q5_1 | 190 | 5.7 | 13.0 | 2.0 s | 370 MB | 44.3 / 28.0 |
| Student, f16 (published) | 488 | 4.9 | 10.9 | 2.4 s | 653 MB | 28.9 / 22.2 |
| Whisper small, f16 | 488 | 4.9 | 10.8 | 2.3 s | 653 MB | 43.2 / 29.0 |
| Whisper large-v3-turbo, q5_0 | 574 | 1.2 | 2.9 | 9.6 s | 760 MB | 15.6 / 13.9 |
| Whisper large-v3-turbo, f16 | 1,624 | 0.9 | 2.2 | 13.0 s | 1,760 MB | 17.2 / 12.6 |

- **q8_0** was 2.6 times as fast as f16 with no accuracy loss visible on these clips. The gain uses the chip's 8-bit instructions, so it will differ on Intel and AMD.
- **Accuracy** is from 30 clips per test, so it only compares models with each other. The 400-clip and 150-clip scores above are the better estimates.
- **Parakeet** ran through its command-line tool, so its speed leaves out startup and any server. It rewrites Serbian in Croatian forms, which count as errors.
- **Stability:** the student was run again at the end and gave the same speeds.

Reproduce it with `cpu_benchmark.py`; raw results are in `results/cpu.tsv`. The q8_0 file is made with `whisper-quantize models/ggml-small-sr-distill.bin models/ggml-small-sr-distill-q8_0.bin q8_0`.

## Three languages

Word error rate in %, lower is better. 100 FLEURS test clips per language, whisper.cpp 1.9.6, language given, Serbian compared in Latin script.

| Model | Serbian | Croatian | English |
|---|---|---|---|
| Whisper small | 38.2 | 35.2 | 7.0 |
| **Student** | **24.6** | **23.6** | 7.5 |
| Parakeet TDT 0.6B v3, q8_0 | 21.2 | 10.9 | 5.8 |
| Whisper large-v3-turbo (teacher) | 12.4 | 11.1 | 5.5 |

Raw numbers are in `results/`: `serbian_tests.txt`, `croatian_tests.txt` and `q8_0_tests.txt` for the tests above, `cpu.tsv` for CPU, `student.tsv` and `baselines.tsv` for this table. Parakeet's Serbian output uses Croatian forms, which this score counts as errors.

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
python speech_tests.py student=models/ggml-small-sr-distill.bin --lang sr
python speech_tests.py student=models/ggml-small-sr-distill.bin --lang hr
python cpu_benchmark.py student=models/ggml-small-sr-distill.bin
```

`speech_tests.py` downloads the parliament file and the FLEURS test set, about 900 MB per language, on its first run.
`prepare.py` downloads about 3 GB in total and keeps 0.8 GB. The teacher pass takes about 24 minutes and is cached in `work/teacher.pt`.
The watermark variables cap GPU memory at about 18 GB on a 36 GB Mac. Without a cap, an early run swapped until the disk filled up.

Use the model with whisper.cpp:

```sh
whisper-cli -m models/ggml-small-sr-distill.bin -l sr -f audio.wav
```

## License

Code: MIT. Weights derive from [whisper-small](https://huggingface.co/openai/whisper-small) (Apache-2.0) and [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT). Training data: [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0).
