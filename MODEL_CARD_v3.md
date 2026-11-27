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

# whisper-small-sr-distill-v3

Whisper small, distilled from Whisper large-v3-turbo for Serbian and Croatian. It is trained on the same data as [v2](https://huggingface.co/rawmind/whisper-small-sr-distill-v2), but on noisy and reverberant versions of the audio, and it keeps the teacher's targets from the clean clip. It has the same size and speed as `small`. Compared with v2, it handles noise better and drops fewer words when audio arrives in streaming windows. In q8_0 it is 264 MB.

**Set the language.** Language detection was unusable in v2, which called most Serbian and Croatian speech English, and v3 was not retested.

**Training code and raw results:** [rawmind/whisper-distill-sr](https://github.com/rawmind/whisper-distill-sr) on GitHub.

## Streaming, as in Titl

[Titl](https://github.com/rawmind/titl) sends 8–12 s windows of a long stream to whisper-server with speech detection on. It keeps the segments that end at least 1 s before the window's end and merges the text. This test runs Titl's own code on 150 held-out parliament clips per language, joined into one stream, and scores the stitched text. Fast feeding gives about 8 s windows, as on a GPU. Backlog gives 12 s windows, as on a slow CPU.

| Stream | Feeding, context | v2 | v3 |
|---|---|---|---|
| Serbian, 30.7 min | fast, full | 22.1 | **18.1** |
| | fast, short | 22.4 | **20.3** |
| | backlog, full | 21.7 | **15.9** |
| | backlog, short | 17.7 | 17.7 |
| Croatian, 28.3 min | fast, full | 28.7 | **24.5** |
| | fast, short | 29.9 | **25.4** |
| | backlog, full | 25.9 | **21.3** |
| | backlog, short | 25.8 | **22.0** |

These are word error rates in %, with q8_0 models and whisper.cpp 1.9.6. Most of the gain comes from fewer dropped words. v2 often skips the first words of a window that starts in the middle of speech.

## Sentence tests

| Test | Clips | Serbian v2 | v3 | Change (95%) | Croatian v2 | v3 | Change (95%) |
|---|---|---|---|---|---|---|---|
| FLEURS, clean read speech | 400 | 17.3 | 17.4 | +0.1 (−0.6 to +0.7) | 18.0 | 18.4 | +0.4 (−0.3 to +1.2) |
| Parliament, real speech | 150 | 10.6 | **9.6** | −0.9 (−1.6 to −0.2) | 15.7 | 16.6 | +0.9 (−0.1 to +1.9) |
| FLEURS + pink noise, 10 dB | 100 | 22.4 | **20.3** | −2.1 (−3.8 to −0.4) | 29.8 | **26.3** | −3.5 (−5.5 to −1.5) |
| FLEURS + pink noise, 5 dB | 100 | 28.7 | **25.8** | −2.8 (−4.5 to −1.3) | 47.5 | **37.7** | −9.9 (−12.4 to −7.2) |
| FLEURS, shortened context | 100 | 16.5 | 16.9 | +0.4 (−0.9 to +1.8) | 18.0 | 19.1 | +1.1 (−1.0 to +3.5) |
| Parliament, shortened context | 100 | 11.4 | 10.9 | −0.5 (−1.4 to +0.4) | 15.2 | 15.4 | +0.2 (−1.0 to +1.6) |

These are word error rates in %, with both models in q8_0 on whisper.cpp 1.9.6 with Metal. The change is measured clip by clip, with a 95% bootstrap range. For reference, large-v3-turbo scores 13.2 and 10.1 on Serbian FLEURS and parliament, and 11.4 and 13.6 on Croatian.

- **Noise:** v3 is clearly better in both languages. The test noise is pink, which was not used in training. Pink sits between the white and brown noise that was used, so this shows robustness to steady synthetic noise rather than to real recordings.
- **Clean speech:** v3 is level with v2 in Serbian and slightly behind in Croatian. On Croatian parliament speech it is worse on 55 clips and better on 36.
- **Shortened context:** whisper.cpp's `audio_ctx` is set to the clip length plus one second, which speeds up short clips on slow machines.

## Read the parliament scores with care

The parliament test clips come from the same kind of recordings as the parliament training data, so these scores are optimistic. Every clip from the test speakers and the test dates was left out of training, as were the test sentences. But the vocabulary, the room acoustics and the style of speech are the same. Expect results closer to the FLEURS scores on other speech, and test on your own audio. The transcripts are official records and not always word for word, so compare models with each other rather than reading the totals as absolute.

## Limits

- **Set the language** (`-l sr` or `-l hr`). Language detection was not retested.
- **It writes text over silence.** On clips with no speech, such as digital silence or plain noise, it writes text, as v2 and turbo do. Use speech detection, such as whisper.cpp's `--vad` with [Silero](https://huggingface.co/ggml-org/whisper-vad).
- **English was not retested.** v2 scored 8.3% on 100 FLEURS clips, against 7.0% for Whisper `small`.
- **Other languages are untested** and are likely worse than Whisper `small`.
- **Untested on music and on several people talking.** The noise tests use artificial pink noise.
- **It was trained once.** A second run with 40% clean clips instead of 20% landed within about 1 point of it on the sentence tests. It was 0.4 to 2.9 points worse in streaming, so small differences between versions may be run-to-run variation.
- **Serbian is written in Latin script.**
- **Use whisper.cpp 1.9.6 or newer.**

## Use

Files: the Transformers weights, and for whisper.cpp `ggml-small-sr-distill-v3.bin` (f16, 488 MB) and `ggml-small-sr-distill-v3-q8_0.bin` (264 MB).

```sh
whisper-server -m ggml-small-sr-distill-v3-q8_0.bin -l sr --vad --vad-model ggml-silero-v6.2.0.bin
```

## Training

- **Data:** the same 20,968 clips as v2: FLEURS Serbian, Croatian and English, plus ParlaSpeech-RS and ParlaSpeech-HR. The tenth of the parliament clips that the teacher found least likely was removed. No FLEURS test sentence is in the training data, and no parliament test speaker, date or sentence.
- **Augmentation:** 20% of clips stay clean. The rest get, at random:
  - a speed change of 0.9–1.1× (30%)
  - synthetic room reverb with RT60 0.2–0.8 s (30%)
  - noise at 0–20 dB SNR (80%): white, brown, babble of 3–6 other training clips, or mains hum
  - a lowpass at 3.4–7 kHz (25%)
  - 8-bit mu-law (10%)

  The teacher's targets always come from the clean clip.
- **Method:** the teacher's 64 most likely next tokens for each clip were stored once. The student learns from the transcript (cross-entropy) and from those tokens (KL divergence, temperature 2), weighted equally.
- **Optimizer:** AdamW, learning rate 2e-5 with cosine decay, 9,759 steps (1.9 epochs) in 4 hours on an Apple M4 Max.

## License and attribution

The weights are released under **CC BY-SA 4.0** because they were trained on ParlaSpeech, which is distributed under that license.

- Derived from [openai/whisper-small](https://huggingface.co/openai/whisper-small) (Apache-2.0), with [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT) as the teacher.
- Trained on [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google, CC BY 4.0).
- Trained on ParlaSpeech-RS ([CLARIN.SI](http://hdl.handle.net/11356/1834), [Hugging Face](https://huggingface.co/datasets/classla/ParlaSpeech-RS)) and ParlaSpeech-HR ([CLARIN.SI](http://hdl.handle.net/11356/1914), [Hugging Face](https://huggingface.co/datasets/classla/ParlaSpeech-HR)), CC BY-SA 4.0. The audio and transcripts were used for training only and are not redistributed. Serbian transcripts were converted to Latin script.
