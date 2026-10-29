// zen-lipsync (browser entry): real-time photoreal lip-sync on WebGPU, from speech audio and an avatar pack.
// The browser entry needs WebGPU (create() rejects with WebGPUUnavailableError without it); for servers and CPUs
// use zen-lipsync/node.
export { create, avatarLayout, type BrowserCreateOptions, type AvatarFit } from './browser/create.js';
export * from './core/index.js';
// Lower-level building blocks of the WebGPU runtime.
export { LipSyncPlayer, AutoPairing, type DrawInfo, type DrawPhase, type PatchSync } from './runtime/player.js';
export { Avatar, Compositor, loadModels, requestGpuAdapter, type LipPatch, type ModelOptions, type SourceFrame, type View } from './runtime/avatar.js';
export { WgslLips, WGSL_MAX_BATCH, dequantFrameI8, lipWeightsLabel } from './runtime/wgsl.js';
export { fetchModel, setModelHashes, MODEL_CACHE } from './runtime/model-cache.js';
export { browserPackSource } from './runtime/pack-source.js';
