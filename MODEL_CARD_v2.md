---
license: cc-by-sa-4.0
language:
- sr
- hr
- en
base_model: openai/whisper-small
pipeline_tag: automatic-speech-recognition
library_name: transformers
datasets:
- google/fleurs
- classla/ParlaSpeech-RS
- classla/ParlaSpeech-HR
tags:
- whisper
- distillation
- serbian
- croatian
- whisper.cpp
---

# whisper-small-sr-distill-v2

Whisper small, distilled from Whisper large-v3-turbo for Serbian and Croatian, trained on read speech and parliament speech. Same size and speed as `small`. In Serbian it comes close to large-v3-turbo on parliament speech and beats it when the context is shortened, at a sixth of the size in q8_0 (264 MB).

**Set the language.** Language detection is unusable: it calls most Serbian and Croatian speech English.

**Training code and raw results:** [rawmind/whisper-distill-sr](https://github.com/rawmind/whisper-distill-sr) on GitHub.

## Serbian tests

| Serbian test | Clips | v2 | v1 ([whisper-small-sr-distill](https://huggingface.co/rawmind/whisper-small-sr-distill)) | large-v3-turbo |
|---|---|---|---|---|
| FLEURS, clean read speech | 400 | **17.4** (16.3–18.5) | 24.6 (23.2–25.9) | 13.2 (12.4–14.2) |
| Parliament, real speech | 150 | **10.3** (9.0–11.9) | 22.1 (20.2–24.0) | 10.1 (8.8–11.5) |
| FLEURS + pink noise, 10 dB | 100 | **22.2** (19.6–24.8) | 29.1 (26.6–31.8) | 16.9 (14.9–19.0) |
| FLEURS + pink noise, 5 dB | 100 | **28.5** (25.5–31.3) | 36.3 (33.2–39.4) | 19.9 (17.7–22.2) |
| FLEURS, shortened context | 100 | **16.4** (14.4–18.5) | 27.9 (24.4–32.2) | 22.1 (16.7–28.2) |
| Parliament, shortened context | 100 | **11.3** (9.6–13.3) | 22.4 (20.0–25.0) | 15.9 (12.9–19.7) |

Word error rate in % (lower is better), whisper.cpp 1.9.6 built from source, Metal, with a 95% bootstrap range over clips. Compared clip by clip, v2 beats v1 in all six tests by 6.9 to 11.8 points. Against turbo it is behind on clean and noisy FLEURS, level on parliament (+0.2, 95% −0.9 to +1.3), and ahead with a shortened context (−5.8 on FLEURS, −4.6 on parliament).

- **Shortened context:** whisper.cpp's `audio_ctx` set to the clip length plus one second, which speeds up short clips on slow machines.
- **Speed:** 253 ms per clip against 262 ms for v1 on an M4 Max GPU, the same as Whisper `small`. Turbo is about 2.3 times slower.

## Croatian tests

| Croatian test | Clips | v2 | v1 | large-v3-turbo |
|---|---|---|---|---|
| FLEURS, clean read speech | 400 | **17.9** (16.8–19.1) | 24.3 (23.0–25.7) | 11.4 (10.4–12.4) |
| Parliament, real speech | 150 | **15.9** (14.1–17.8) | 29.6 (27.1–31.8) | 13.6 (11.9–15.4) |
| FLEURS + pink noise, 10 dB | 100 | **29.6** (26.6–32.4) | 37.9 (35.3–40.7) | 17.7 (15.4–19.9) |
| FLEURS + pink noise, 5 dB | 100 | **47.3** (43.8–51.0) | 52.6 (49.2–56.1) | 29.9 (26.9–33.1) |
| FLEURS, shortened context | 100 | **18.1** (15.9–20.3) | 25.8 (23.0–28.9) | 16.7 (12.5–21.3) |
| Parliament, shortened context | 100 | **15.4** (13.1–17.7) | 30.0 (27.2–32.9) | 17.4 (14.3–20.7) |

Same units and method. v2 beats v1 in all six by 5.4 to 14.6 points. Turbo stays ahead in Croatian except with a shortened context, where the two are level.

## Read the parliament scores with care

The parliament test clips come from the same kind of recordings as the parliament training data, so these scores are optimistic. Every clip from the test speakers and the test dates was left out of training, as were the test sentences, but the vocabulary, the room acoustics and the style of speech are the same. Expect results closer to the FLEURS scores on other speech, and test on your own audio.

The parliament transcripts are official records and not always word for word, so compare models with each other rather than reading the totals as absolute.

## Quantized q8_0

`ggml-small-sr-distill-v2-q8_0.bin` is 264 MB instead of 488 MB and scored within 0.6 points of f16 in every test that ran both on the same build. On v1, q8_0 was 2.6 times as fast as f16 on an M4 Max CPU and used 443–497 MB of memory instead of 653 MB; v2 has the same architecture. Speed on Intel and AMD is untested. On CPU, start with q8_0.

## Limits

- **Language detection is unusable.** On 50 clips per language it answered English for 41 Serbian and 33 Croatian clips, and Serbian for none. Always pass the language (`-l sr` or `-l hr`).
- **It writes text over silence.** On five clips with no speech (digital silence and plain noise) it wrote text every time in both languages, for example `„` or mixed-script gibberish. Use speech detection, such as whisper.cpp's `--vad` with [Silero](https://huggingface.co/ggml-org/whisper-vad).
- **English is slightly worse:** 8.3% against 7.5% for v1 and 7.0% for Whisper `small` (100 FLEURS clips).
- **Other languages are untested** and are likely worse than Whisper `small`.
- **Untested on music, video and several people talking.** The noise tests use artificial pink noise.
- **Serbian is written in Latin script.** Serbian training transcripts were converted to Latin.
- **Use whisper.cpp 1.9.6 or newer.** With a shortened context, 1.9.5 made 2.8 to 4.1 more errors per 100 words on the same clips.

## Use

Files: the Transformers weights, and for whisper.cpp `ggml-small-sr-distill-v2.bin` (f16, 488 MB) and `ggml-small-sr-distill-v2-q8_0.bin` (264 MB).

```sh
whisper-cli -m ggml-small-sr-distill-v2-q8_0.bin -l sr -f audio.wav
```

With speech detection:

```sh
whisper-server -m ggml-small-sr-distill-v2-q8_0.bin -l sr --vad --vad-model ggml-silero-v6.2.0.bin
```

## Training

- **Data:** 22,885 clips: FLEURS Serbian (2,944), Croatian (377) and English (394) training clips, plus 15,248 Serbian clips from ParlaSpeech-RS and 3,922 Croatian clips from ParlaSpeech-HR. The tenth of the parliament clips that the teacher found least likely given their transcript (1,917 clips, to remove transcripts that don't match the audio) was dropped, leaving 20,968. No FLEURS test sentence and no parliament test speaker, date or sentence is in the training data.
- **Method:** the teacher reads the human transcript of each clip and its 64 most likely next tokens are stored once. The student learns from the transcript (cross-entropy) and from those tokens (KL divergence, temperature 2), weighted equally.
- **Optimizer:** AdamW, learning rate 2e-5 with cosine decay, 10,354 steps (2.0 epochs) in 4 hours on an Apple M4 Max, after a 2.3-hour teacher pass.

## License and attribution

The weights are released under **CC BY-SA 4.0** because they were trained on ParlaSpeech, which is distributed under that license.

- Derived from [openai/whisper-small](https://huggingface.co/openai/whisper-small) (Apache-2.0), with [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT) as the teacher.
- Trained on [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google, CC BY 4.0).
- Trained on ParlaSpeech-RS ([CLARIN.SI](http://hdl.handle.net/11356/1834), [Hugging Face](https://huggingface.co/datasets/classla/ParlaSpeech-RS)) and ParlaSpeech-HR ([CLARIN.SI](http://hdl.handle.net/11356/1914), [Hugging Face](https://huggingface.co/datasets/classla/ParlaSpeech-HR)), CC BY-SA 4.0. The audio and transcripts were used for training only and are not redistributed; Serbian transcripts were converted to Latin script and clips were filtered as described above.
