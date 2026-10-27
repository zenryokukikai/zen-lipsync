// Errors the SDK throws, shared by both entries.

/** The browser entry needs WebGPU: navigator.gpu is missing, or no GPU adapter was given. There is no CPU fallback in the browser. */
export class WebGPUUnavailableError extends Error {
  override readonly name = 'WebGPUUnavailableError';
  constructor(readonly reason: 'no-navigator-gpu' | 'no-adapter', message?: string) {
    super(message ?? (reason === 'no-navigator-gpu'
      ? 'WebGPU is not available in this browser (navigator.gpu is missing). ZEN-LipSync needs WebGPU in the browser.'
      : 'WebGPU is available but no GPU adapter was found. ZEN-LipSync needs WebGPU in the browser.'));
  }
}

/** The options given to create() are not valid. */
export class OptionsError extends Error {
  override readonly name = 'OptionsError';
}

/** The avatar or a pack is missing a file or is not a ZEN-LipSync avatar pack (version 1). */
export class AvatarPackError extends Error {
  override readonly name = 'AvatarPackError';
}

/** A part of the SDK that is not built yet. */
export class NotImplementedError extends Error {
  override readonly name = 'NotImplementedError';
  constructor(what: string) { super(`${what}: not implemented yet`); }
}
