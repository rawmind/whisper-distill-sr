# whisper-distill-sr

Distils Whisper large-v3-turbo into Whisper small for Serbian speech, on one Apple Silicon Mac.
The student keeps the size and speed of `small` (488 MB in whisper.cpp format).

**Download the model:** [rawmind/whisper-small-sr-distill](https://huggingface.co/rawmind/whisper-small-sr-distill) on Hugging Face.

## Results

Word error rate in %, lower is better. 100 FLEURS test clips per language, whisper.cpp 1.9.6, language given, Serbian compared in Latin script.

| Model | Serbian | Croatian | English |
|---|---|---|---|
| Whisper small | 38.2 | 35.2 | 7.0 |
| **Student** | **24.6** | **23.6** | 7.5 |
| Parakeet TDT 0.6B v3, q8_0 | 21.2 | 10.9 | 5.8 |
| Whisper large-v3-turbo (teacher) | 12.4 | 11.1 | 5.5 |

Raw numbers are in `results/`. Parakeet's Serbian output uses Croatian forms, which this score counts as errors.

## Limits

- Trained and tested on FLEURS, clean read speech. Not tested on other audio, such as video, noise or music.
- Trained on Serbian (2,944 clips), Croatian (200) and English (200). Other languages are untested and may be worse than `small`.
- Language detection is poor: on 50 Serbian clips it answered Croatian 35 times and Serbian 14. Set the language.
- Each language's test is 100 clips, so a difference of about 1 point means nothing.

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
```

`prepare.py` downloads about 3 GB in total and keeps 0.8 GB. The teacher pass takes about 24 minutes and is cached in `work/teacher.pt`.
The watermark variables cap GPU memory at about 18 GB on a 36 GB Mac. Without a cap, an early run swapped until the disk filled up.

Use the model with whisper.cpp:

```sh
whisper-cli -m models/ggml-small-sr-distill.bin -l sr -f audio.wav
```

## License

Code: MIT. Weights derive from [whisper-small](https://huggingface.co/openai/whisper-small) (Apache-2.0) and [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT). Training data: [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0).
