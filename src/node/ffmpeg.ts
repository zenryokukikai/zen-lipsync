// ffmpeg (an optional external program): decodes the idle videos and encodes the outputs. Found on PATH, or given by
// the ffmpegPath option / the ZEN_LIPSYNC_FFMPEG environment variable.
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';

export class FfmpegMissingError extends Error {
  override readonly name = 'FfmpegMissingError';
  constructor(path: string) {
    super(`ffmpeg was not found ("${path}"). ZEN-LipSync on Node needs ffmpeg for the idle video and the video outputs: install it (e.g. apt install ffmpeg) or pass { ffmpegPath }.`);
  }
}

const checked = new Map<string, boolean>();

/** The ffmpeg command to use; throws FfmpegMissingError when it does not run. */
export function resolveFfmpeg(path?: string): string {
  const p = path ?? process.env['ZEN_LIPSYNC_FFMPEG'] ?? 'ffmpeg';
  let ok = checked.get(p);
  if (ok === undefined) {
    const r = spawnSync(p, ['-hide_banner', '-version'], { stdio: 'ignore' });
    ok = r.status === 0;
    checked.set(p, ok);
  }
  if (!ok) throw new FfmpegMissingError(p);
  return p;
}

/** Runs ffmpeg and collects its stdout (for decoding). */
export function ffmpegOutput(ffmpeg: string, args: string[]): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const p = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
    const out: Buffer[] = []; const err: Buffer[] = [];
    p.stdout.on('data', (d: Buffer) => out.push(d));
    p.stderr.on('data', (d: Buffer) => err.push(d));
    p.on('error', reject);
    p.on('close', (code) => code === 0 ? resolve(Buffer.concat(out)) : reject(new Error(`ffmpeg exited with ${code}: ${Buffer.concat(err).toString().slice(-500)}`)));
  });
}

/** Spawns an encoder: video frames (rgb24) on stdin, audio (f32le mono) on fd 3. */
export function spawnEncoder(ffmpeg: string, args: string[]): ChildProcess {
  return spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', ...args], { stdio: ['pipe', 'pipe', 'pipe', 'pipe'] });
}
