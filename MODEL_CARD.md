---
license: apache-2.0
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

- Trained and tested on FLEURS, clean read speech. Not tested on other audio, such as video, noise or music.
- Trained on 2,944 Serbian, 200 Croatian and 200 English clips. Other languages are untested and may be worse than `small`.
- Language detection is poor: on 50 Serbian clips it answered Croatian 35 times and Serbian 14.
- The test is 100 clips per language, so a difference of about 1 point means nothing.

## Training

Cross-entropy on the FLEURS human transcripts and KL divergence to the teacher's 64 most likely tokens (temperature 2), weighted equally. AdamW at 2e-5 with cosine decay, 1,353 steps of batch 4 (1.6 epochs) on an Apple M4 Max.

Derived from [whisper-small](https://huggingface.co/openai/whisper-small) (Apache-2.0), with [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT) as the teacher. Trained on [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0).
