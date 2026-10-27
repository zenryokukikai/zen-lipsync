// A worker thread preparing speech chunks (resampling to 16 kHz, log-mel features, the CPU sentence model), so this
// work does not hold up the frames being drawn (Node: worker_threads).
// Messages: { id, pcm, sampleRate, ref? } -> { id, melFrames, frames, gather } (features only, when ref is absent)
//           { id, mel, ref, T } -> { id, z } (sentence model) ; errors: { id, error }.
import { parentPort, workerData } from 'node:worker_threads';
import { logMel, MEL_SAMPLE_RATE } from '../core/mel.js';
import { resample } from '../core/resample.js';
import { CpuSentenceModel, parseSentenceWeights } from '../core/sentence-model.js';

const model = new CpuSentenceModel(parseSentenceWeights((workerData as { weights: ArrayBuffer }).weights));
parentPort!.on('message', (m: { id: number; kind: 'mel'; pcm: Float32Array; sampleRate: number } | { id: number; kind: 'sentence'; mel: Float32Array; ref: Float32Array; T: number }) => {
  try {
    if (m.kind === 'mel') {
      const pcm16 = m.sampleRate === MEL_SAMPLE_RATE ? m.pcm : resample(m.pcm, m.sampleRate, MEL_SAMPLE_RATE);
      const r = logMel(pcm16);
      parentPort!.postMessage({ id: m.id, mel: r.mel, melFrames: r.melFrames, frames: r.frames, gather: r.gather }, [r.mel.buffer as ArrayBuffer]);
    } else {
      const z = model.run(m.mel, m.ref, m.T);
      parentPort!.postMessage({ id: m.id, z }, [z.buffer as ArrayBuffer]);
    }
  } catch (e) {
    parentPort!.postMessage({ id: m.id, error: String(e) });
  }
});
