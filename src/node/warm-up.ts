// One run of the speech steps (resample, log-mel, the CPU sentence model) when a worker starts, results dropped: the
// first speech otherwise pays V8 compiling them. Used by the Node speech worker.
import { logMel, MEL_SAMPLE_RATE } from '../core/mel.js';
import { resample } from '../core/resample.js';
import type { CpuSentenceModel } from '../core/sentence-model.js';

export function warmSpeechSteps(model: CpuSentenceModel): void {
  const pcm = new Float32Array(44100).map((_, i) => Math.sin(i * 0.05) * 0.1);
  const r = logMel(resample(pcm, 44100, MEL_SAMPLE_RATE));
  model.run(r.mel, new Float32Array(r.melFrames * 128), r.melFrames);
}
