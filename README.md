# whisper-distill-sr

Distils Whisper large-v3-turbo into Whisper small for Serbian speech, on one Apple Silicon Mac.
The student keeps the size and speed of `small` (488 MB in whisper.cpp format).

**Download the model:** [rawmind/whisper-small-sr-distill](https://huggingface.co/rawmind/whisper-small-sr-distill) on Hugging Face. A second version trained on parliament speech too, with fewer errors, is [rawmind/whisper-small-sr-distill-v2](https://huggingface.co/rawmind/whisper-small-sr-distill-v2) (CC BY-SA 4.0, see [below](#more-training-data-student_improved)).

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

`ggml-small-sr-distill-q8_0.bin` is 264 MB instead of 488 MB and scores within 0.6 points of f16 in all these tests (word error rate in %, same clips as above):

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
| Whisper large-v3-turbo, q8_0 | 874 | 3.4 | 9.1 | 3.4 s | 1,050 MB | 17.0 / 12.5 |
| Whisper large-v3-turbo, q5_0 | 574 | 1.2 | 2.9 | 9.6 s | 760 MB | 15.6 / 13.9 |
| Whisper large-v3-turbo, f16 | 1,624 | 0.9 | 2.2 | 13.0 s | 1,760 MB | 17.2 / 12.6 |

- **q8_0** was 2.6 times as fast as f16 with no accuracy loss visible on these clips. The gain uses the chip's 8-bit instructions, so it will differ on Intel and AMD.
- **Accuracy** is from 30 clips per test, so it only compares models with each other. The 400-clip and 150-clip scores above are the better estimates.
- **Parakeet** ran through its command-line tool, so its speed leaves out startup and any server. It rewrites Serbian in Croatian forms, which count as errors.
- **Stability:** the student was run again at the end and gave the same speeds. Turbo q8_0 comes from a later run, in which the student and `small` q8_0 gave the same speeds and error rates as before.
- **Turbo q8_0** was 2.8 times as fast as q5_0. About 3.4 times real time on an M4 Max may not be enough on a slower laptop.

Reproduce it with `cpu_benchmark.py`; raw results are in `results/cpu.tsv` and `results/cpu_turbo_q8_vad.tsv`. The q8_0 file is made with `whisper-quantize models/ggml-small-sr-distill.bin models/ggml-small-sr-distill-q8_0.bin q8_0`.

## Voice detection (VAD)

whisper-server can skip non-speech audio with `--vad --vad-model ggml-silero-v6.2.0.bin` ([Silero](https://huggingface.co/ggml-org/whisper-vad), 0.9 MB). Same CPU and 60 clips as above, full context:

| Model | No-speech clips with text | WER: FLEURS / parliament | Speed |
|---|---|---|---|
| Student q8_0 | 5 of 5 | 28.7 / 21.5 | 12.7 |
| Student q8_0 + VAD | **0 of 5** | 27.1 / 21.7 | 12.4 |
| Whisper small q8_0 | 5 of 5 | 41.6 / 28.2 | 12.7 |
| Whisper small q8_0 + VAD | **0 of 5** | 43.4 / 28.7 | 12.2 |
| Whisper large-v3-turbo q8_0 | 5 of 5 | 17.0 / 12.5 | 3.4 |
| Whisper large-v3-turbo q8_0 + VAD | **0 of 5** | 15.9 / 13.0 | 3.3 |

**Subtitle timing** was tested on 60 made-up clips of each kind with known speech times: one sentence in an 8 s window, and two sentences with a 1.5–4 s gap. Segment times with the Metal build, in seconds:

| Model | Clips | Starts within 0.5 s | Median start error | Ends within 0.5 s | Subtitle over silence |
|---|---|---|---|---|---|
| Student q8_0 | window | 0% → **100%** | 1.98 → **0.03** | 88% → **100%** | 61% → **0%** |
| Student q8_0 | two sentences | 30% → **52%** | 1.37 → **0.10** | 82% → **99%** | 51% → 44% |
| large-v3-turbo f16 | window | 3% → **100%** | 1.93 → **0.03** | 67% → **100%** | 73% → **0%** |
| large-v3-turbo f16 | two sentences | 32% → **61%** | 1.21 → **0.04** | 92% → **99%** | 38% → 21% |

Each cell is without VAD → with VAD. Without it, segments usually start in the silence before the speech, about 1–2 s early. With VAD two sentences can still share one segment, so 39–48% of sentence starts were more than 0.5 s off. VAD also left a little more speech without a subtitle in the window clips (2–5% → 4–7%).

- The no-speech clips are digital silence and plain noise, and the gaps are low noise. Music and background talk are untested and may still get through.
- Accuracy differences between VAD and no VAD are within the noise of 30 clips per test.

Reproduce with `cpu_benchmark.py --silence --server-args "--vad --vad-model ..."` and `timing_test.py`; raw results are in `results/vad_timing.txt`.

## More training data: student_improved

A second student, trained on 20,968 clips instead of 3,344: 3,715 FLEURS clips (2,944 Serbian, 377 Croatian, 394 English), plus 15,248 Serbian and 3,922 Croatian clips from [ParlaSpeech-RS](https://huggingface.co/datasets/classla/ParlaSpeech-RS) and [ParlaSpeech-HR](https://huggingface.co/datasets/classla/ParlaSpeech-HR), minus the 1,917 parliament clips (a tenth) that the teacher found least likely. Same size and architecture. Training took 10,354 steps (2.0 epochs) in 4 hours, after a 2.3-hour teacher pass.

**Download:** [rawmind/whisper-small-sr-distill-v2](https://huggingface.co/rawmind/whisper-small-sr-distill-v2) on Hugging Face, as `ggml-small-sr-distill-v2.bin` (f16) and `ggml-small-sr-distill-v2-q8_0.bin`, with the Transformers weights. The weights are CC BY-SA 4.0 because of the ParlaSpeech data, unlike the MIT of the first student. Its model card is [MODEL_CARD_v2.md](MODEL_CARD_v2.md).

Word error rate in %, whisper.cpp 1.9.6 built from source, same clips as the tables above. The change is against the published student, clip by clip, with a 95% range.

| Test | Clips | Serbian | Change | Turbo | Croatian | Change | Turbo |
|---|---|---|---|---|---|---|---|
| FLEURS | 400 | **17.4** | −7.2 (−8.2 to −6.2) | 13.2 | **17.9** | −6.4 (−7.3 to −5.5) | 11.4 |
| Parliament | 150 | **10.3** | −11.8 (−13.4 to −10.2) | 10.1 | **15.9** | −13.7 (−15.2 to −12.2) | 13.6 |
| FLEURS + pink noise, 10 dB | 100 | **22.2** | −6.9 (−9.1 to −4.9) | 16.9 | **29.6** | −8.4 (−10.7 to −6.3) | 17.7 |
| FLEURS + pink noise, 5 dB | 100 | **28.5** | −7.9 (−10.1 to −5.8) | 19.9 | **47.3** | −5.4 (−8.1 to −2.7) | 29.9 |
| FLEURS, shortened context | 100 | **16.4** | −11.6 (−15.5 to −8.4) | 22.1 | **18.1** | −7.7 (−10.5 to −5.5) | 16.7 |
| Parliament, shortened context | 100 | **11.3** | −11.1 (−12.9 to −9.4) | 15.9 | **15.4** | −14.6 (−16.9 to −12.5) | 17.4 |

The q8_0 file scored within 0.6 points of f16 in every test where both ran on the same build.

- **Parliament scores are optimistic:** the parliament test files are of the same kind as the training files. Every clip from the test speakers and test dates was excluded from training, as were the test sentences.
- **Language detection is worse than the published student's, and unusable.** On 50 clips per language it called 41 of 50 Serbian clips English and 33 of 50 Croatian ones, and none of the Serbian ones Serbian. Set the language.
- **English** is 8.3% against 7.5% for the published student on 100 clips.
- **Silence:** it writes text on all five Serbian and all five Croatian no-speech clips. Use speech detection.
- **Music and several people talking** are untested, as before.
- **Licenses:** the master copies of both ParlaSpeech datasets, on CLARIN.SI ([HR](http://hdl.handle.net/11356/1914), [RS](http://hdl.handle.net/11356/1834)), are CC BY-SA 4.0. Hugging Face lists ParlaSpeech-RS as CC BY 4.0 and no license for ParlaSpeech-HR.

Prepare the data with `prepare_parliament.py` (about 5 GB of downloads, 4 GB kept) and train with `distill.py --work work/more --hr 400 --en 400 --minutes 240 --parliament`. The training log is `results/train_student_improved.log`, the raw scores are in `results/student_improved.txt`.

## A larger teacher: large-v3

Would full Whisper large-v3 (32 decoder layers, 3.1 GB) teach better than large-v3-turbo (4 decoder layers)? Both ran on the same clips with the same method, whisper.cpp 1.9.6 from Homebrew on an M4 Max GPU. Turbo was run again on this build and matched the tables above to the decimal.

Word error rate in %, with a 95% bootstrap range over clips.

| Test | Clips | Serbian large-v3 | Serbian turbo | Croatian large-v3 | Croatian turbo |
|---|---|---|---|---|---|
| FLEURS | 400 | 13.0 (12.1–13.9) | 13.2 (12.4–14.2) | **10.7** (9.8–11.7) | 11.4 (10.4–12.4) |
| Parliament | 150 | **9.1** (7.9–10.4) | 10.1 (8.8–11.5) | **12.3** (10.7–14.2) | 13.6 (11.9–15.4) |
| FLEURS + pink noise, 10 dB | 100 | 16.9 (15.1–19.0) | 16.9 (14.9–19.0) | 17.1 (14.8–19.3) | 17.7 (15.4–19.9) |
| FLEURS + pink noise, 5 dB | 100 | 20.1 (18.0–22.4) | 19.9 (17.7–22.2) | **27.4** (24.3–30.3) | 29.9 (26.9–33.1) |
| FLEURS, shortened context | 100 | 22.3 (15.5–31.2) | 22.1 (16.7–28.2) | 17.7 (12.5–24.0) | 16.7 (12.5–21.3) |
| Parliament, shortened context | 100 | 15.1 (11.0–20.8) | 15.9 (12.9–19.7) | 17.9 (13.3–24.1) | 17.4 (14.3–20.7) |

- **Serbian:** a tie. The largest gap is 1.0 point on parliament, and every range overlaps.
- **Croatian:** large-v3 is ahead in four of six tests, by 0.6 to 2.5 points, most in 5 dB noise. The ranges overlap; the two were not compared clip by clip.
- **Serbian forms in Croatian:** 6 of 1,189 words with *je* on FLEURS for large-v3 against 8 for turbo, and 2 of 454 for both on parliament.
- **Silence:** both write text on all five no-speech clips. Turbo writes `Hvala.` or `Hvala što pratite kanal.`; in Serbian, large-v3 also wrote `Продавање` in Cyrillic.
- **Speed** was not measured here.

Turbo stays the teacher: large-v3 offers no headroom for Serbian, and at most a small one for Croatian. More and noisier training data is the likelier way to close the gap to turbo.

The model is `ggml-large-v3.bin` from [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) (SHA-256 `64d182b4…94d1e2`). Run `python speech_tests.py large-v3=models/ggml-large-v3.bin turbo=path/to/ggml-large-v3-turbo.bin --lang sr`, and again with `--lang hr`. The raw scores are in `results/large_v3_sr.txt` and `results/large_v3_hr.txt`.

## whisper.cpp versions

The numbers above are from whisper.cpp 1.9.6. Version matters with a shortened context: 1.9.5 made 2.7 to 21.9 more errors per 100 words than 1.9.6 across three models and both languages, with both built from source on the same clips. At full length it was 0.6 to 1.5 points worse. Homebrew's 1.9.5_2 was far worse on shortened context too. `results/whisper_versions.tsv` has every score with its version.

| Error points 1.9.5 adds over 1.9.6 (95% range) | Serbian FLEURS | Serbian parliament | Croatian FLEURS | Croatian parliament |
|---|---|---|---|---|
| Published student, shortened | +5.1 (0.8 to 9.5) | +10.0 (2.5 to 19.9) | +21.9 (7.7 to 39.0) | +4.9 (1.4 to 10.4) |
| student_improved, shortened | +3.7 (1.9 to 5.7) | +3.2 (2.1 to 4.3) | +4.1 (1.7 to 6.9) | +2.8 (1.2 to 4.6) |
| student_improved q8_0, shortened | +3.1 (1.7 to 4.5) | +3.3 (2.2 to 4.4) | +4.0 (1.6 to 7.1) | +2.7 (1.1 to 4.5) |

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
python timing_test.py student=models/ggml-small-sr-distill.bin --vad-model ggml-silero-v6.2.0.bin
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
