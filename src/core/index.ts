// Shared by every entry (browser, Node): the API types, errors, option checks, avatar pack formats and loading,
// audio features and the lip-sync computation around the models.
export * from './types.js';
export { VERSION } from './version.js';
export { WebGPUUnavailableError, OptionsError, AvatarPackError, NotImplementedError } from './errors.js';
export { resolveOptions, joinAssets, type ResolvedOptions } from './options.js';
export { Emitter } from './events.js';
export { listAvatar, loadPackData, checkPackMeta, checkAvatarSet, fileHash, modelPath, sha256Hex, MODEL_FILES,
  type PackSource, type PackRef, type PackMeta, type PackData, type AvatarSet, type AvatarListing, type Geometry, type ToneTables } from './pack.js';
export { logMel, MEL_SAMPLE_RATE, type LogMel } from './mel.js';
export { resample } from './resample.js';
export { chooseIdlePair, IDLE_GOOD_PAIR_DIST, type IdlePairTable, type IdleSwitchFile, type IdlePairChoice } from './idle-switch.js';
export { sentenceLatents, inferFrames, speechMask, type LipModels, type LipAvatar, type SentenceInput } from './lipsync.js';
export { CpuSentenceModel, parseSentenceWeights, type SentenceWeights } from './sentence-model.js';
