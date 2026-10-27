// Audio file -> video file (long recordings welcome). Run after `pnpm build`:
//   node examples/render-audio.mjs marine /path/to/zen-lipsync-avatars in.wav out.mp4
// The WAV is read in chunks and never held in memory as a whole; the video is written by ffmpeg.
import { renderAudioFile } from '../dist/node/index.js';

const [avatar = 'marine', assetsPath, audio, out] = process.argv.slice(2);
if (!assetsPath || !audio || !out) {
  console.error('usage: node examples/render-audio.mjs <avatar> <assetsPath> <in.wav> <out.mp4>');
  process.exit(2);
}

const r = await renderAudioFile({
  avatar,
  assetsPath,
  audio,
  out,
  crf: 20,
  idleBeforeMs: 500,
  idleAfterMs: 500,
  onProgress: (p) => {
    const pct = p.ratio === null ? '' : ` ${(p.ratio * 100).toFixed(0)}%`;
    console.log(`${(p.timeMs / 1000).toFixed(1)} s of video${pct} (${p.fps.toFixed(1)} fps)`);
  },
});
console.log(`${r.file}: ${(r.durationMs / 1000).toFixed(2)} s, ${r.frames} frames, made in ${(r.wallMs / 1000).toFixed(1)} s (RTF ${r.rtf.toFixed(2)})`);
