import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
  options: {
    titl: { type: 'string' },
    server: { type: 'string' },
    audio: { type: 'string' },
    language: { type: 'string', default: 'sr' },
    mode: { type: 'string', default: 'fast' },
    short: { type: 'boolean', default: false },
  },
});

const load = (file) => import(pathToFileURL(join(values.titl, 'src', file)).href);
const { LookaheadTranscriber } = await load('lookahead.js');
const { Pace, SAMPLE_RATE, transcribe } = await load('whisper.js');
const { DEFAULTS } = await load('settings.js');

const raw = readFileSync(values.audio);
const audio = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const config = { ...DEFAULTS, language: values.language, serverUrl: values.server, requestTimeoutSec: 120 };
const pace = new Pace();
pace.slow = values.short;

const cues = [];
const windows = [];
const errors = [];
const transcriber = new LookaheadTranscriber(config, {
  transcribe: async (samples, language) => {
    const started = performance.now();
    const result = await transcribe(config, samples, { verbose: true, language, pace });
    windows.push({ seconds: samples.length / SAMPLE_RATE, ms: Math.round(performance.now() - started) });
    return result;
  },
  onCues: (found) => cues.push(...found.map(({ start, end, text }) => ({ start, end, text }))),
  onError: (err) => errors.push(err.message),
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const idle = async () => {
  while (transcriber.busy) await sleep(2);
};

const block = SAMPLE_RATE;
for (let start = 0; start < audio.length; start += block) {
  transcriber.addAudio({ t0: start / SAMPLE_RATE, rate: 1, samples: audio.subarray(start, start + block) });
  if (values.mode === 'fast') await idle();
}
transcriber.finish();
await idle();

process.stdout.write(
  JSON.stringify({
    hyp: cues.map((c) => c.text).join(' '),
    cues,
    windows,
    errors,
    stats: transcriber.stats,
  }),
);
