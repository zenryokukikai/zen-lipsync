# ZEN-LipSync SDK API（1.1.0）

組み込み方は [GUIDE.md](GUIDE.md)。この一覧は `node scripts/gen-api-md.mjs` で、型定義から作っています。

## ブラウザ（WebGPU）

`@zenryokukikai/zen-lipsync`

| 名前 | 形 | 説明 |
|---|---|---|
| `create` | <code>create(options: BrowserCreateOptions): Promise&lt;AvatarHandle&gt;</code> | アバターを読み込み、待機映像を流し始めます。 |
| `BrowserCreateOptions` | <code>CreateOptions + { container, fit? }</code> | ブラウザの create() のオプション。 |
| `CreateOptions` | <code>{ avatar, assetsBaseUrl?, weights?, idle?, onProgress? }</code> | create() のオプション。 |
| `AvatarFit` | <code>AvatarFit = 'contain' \| 'cover'</code> | 枠への収め方（既定 contain）。 |
| `AvatarHandle` | <code>{ backend, idle, idleLoops, speaking, speak(), speakStream(), setIdle(), on(), stop(), pause(), resume(), frameTiming(), destroy() }</code> | 読み込んだアバター。 |
| `SpeechAudio` | <code>{ pcm, sampleRate }</code> | 音声：モノラルの Float32Array とサンプリングレート。 |
| `AvatarEvents` | <code>{ ready, speechstart, speechend, idlechange, error }</code> | on() のイベント。 |
| `LoadProgress` | <code>{ loadedBytes, file }</code> | 読み込みの進み具合。 |
| `FrameTiming` | <code>{ frames, inferMsPerFrame }</code> | 1 フレームの推論時間。 |
| `LipWeights` | <code>LipWeights = 'f32' \| 'i8' \| 'i8q'</code> | モデルの重みの形式（既定 i8）。 |
| `WebGPUUnavailableError` | <code>Error + { reason }</code> | WebGPU が使えないとき。 |
| `OptionsError` | <code>extends Error</code> | オプションが正しくないとき。 |
| `AvatarPackError` | <code>extends Error</code> | アバターのファイルを読めないとき。 |
| `VERSION` | <code>const VERSION = "1.1.0";</code> | SDK のバージョン。 |

<details><summary>低レベル</summary>

`AutoPairing`、`Avatar`、`AvatarEventName`、`AvatarListing`、`AvatarSet`、`Compositor`、`CpuSentenceModel`、`DrawInfo`、`DrawPhase`、`Emitter`、`FRAMES_PACK_MAGIC`、`FramesPackIndex`、`Geometry`、`IDLE_GOOD_PAIR_DIST`、`IdlePairChoice`、`IdlePairTable`、`IdleSwitchFile`、`LipAvatar`、`LipModels`、`LipPatch`、`LipSyncPlayer`、`LogMel`、`MEL_SAMPLE_RATE`、`MODEL_CACHE`、`MODEL_FILES`、`ModelOptions`、`NotImplementedError`、`PackData`、`PackMeta`、`PackRef`、`PackSource`、`PatchSync`、`ResolvedOptions`、`SentenceInput`、`SentenceWeights`、`SourceFrame`、`ToneTables`、`View`、`WGSL_MAX_BATCH`、`WgslLips`、`avatarLayout`、`browserPackSource`、`checkAvatarSet`、`checkPackMeta`、`chooseIdlePair`、`dequantFrameI8`、`fetchModel`、`fileHash`、`framesPackFrameEnd`、`framesPackHeadLength`、`inferFrames`、`joinAssets`、`lipWeightsLabel`、`listAvatar`、`loadModels`、`loadPackData`、`logMel`、`modelPath`、`parseSentenceWeights`、`readFramesPackIndex`、`requestGpuAdapter`、`resample`、`resolveOptions`、`sentenceLatents`、`setModelHashes`、`sha256Hex`、`speechMask`

</details>

## React

`@zenryokukikai/zen-lipsync/react`

| 名前 | 形 | 説明 |
|---|---|---|
| `ZenLipSync` | <code>ZenLipSync(props: ZenLipSyncProps): ReactElement</code> | アバターを置くコンポーネント。 |
| `ZenLipSyncProps` | <code>Omit&lt;BrowserCreateOptions, 'container'&gt; + { onReady?, onError?, className?, style? }</code> | props。 |

<details><summary>低レベル</summary>

`AvatarHandle`、`WebGPUUnavailableError`

</details>

## Node.js（CPU）

`@zenryokukikai/zen-lipsync/node`

| 名前 | 形 | 説明 |
|---|---|---|
| `create` | <code>create(options: NodeCreateOptions): Promise&lt;NodeAvatar&gt;</code> | アバターを CPU に読み込みます（ffmpeg が必要）。 |
| `NodeCreateOptions` | <code>{ avatar, assetsPath?, assetsBaseUrl?, backend?, precision?, idle?, ffmpegPath?, idleCache?, onProgress? }</code> | create() のオプション。 |
| `NodeAvatar` | <code>{ kernel, backend, idle, idleLoops, speaking, width, height, frames(), render(), stream(), setIdle(), speak(), speakStream(), on, stop(), pause(), resume(), frameTiming(), destroy() }</code> | 音声から frames()・render()・stream() を作ります。 |
| `FramesOptions` | <code>{ idleBeforeMs?, idleAfterMs?, realtime?, startDelayMs?, audioSampleRate? }</code> | frames() などの共通オプション。 |
| `RenderOptions` | <code>Omit&lt;FramesOptions, 'realtime'&gt; + { out, crf?, onProgress?, signal?, expectedFrames? }</code> | render() のオプション。 |
| `StreamOptions` | <code>FramesOptions + { format, width?, crf? }</code> | stream() のオプション。 |
| `NodeFrame` | <code>{ index, timeMs, width, height, rgb, speaking, audio, sampleRate }</code> | frames() の 1 フレーム。 |
| `RenderProgress` | <code>{ frames, timeMs, totalFrames }</code> | render() の進み具合。 |
| `renderAudioFile` | <code>renderAudioFile(opts: RenderAudioFileOptions): Promise&lt;RenderAudioResult&gt;</code> | WAV ファイル → 動画ファイル。 |
| `RenderAudioFileOptions` | <code>RenderAudioOptions + { avatar, assetsPath?, assetsBaseUrl?, backend?, precision?, idle?, ffmpegPath?, idleCache?, audio, chunkSeconds? }</code> | renderAudioFile() のオプション。 |
| `RenderAudioResult` | <code>{ file, frames, durationMs, audioMs, wallMs, fps, rtf }</code> | renderAudioFile() の結果。 |
| `renderAudio` | <code>renderAudio(avatar, input, opts)</code> | 読み込み済みのアバターで音声 → 動画。 |
| `openWav` | <code>openWav(path: string): Promise&lt;WavReader&gt;</code> | WAV を少しずつ読みます。 |
| `WavReader` | <code>WavInfo + { path, read(), close() }</code> | openWav() の戻り値。 |
| `FfmpegMissingError` | <code>extends Error</code> | ffmpeg がないとき。 |
| `WavFormatError` | <code>extends Error</code> | 読めない WAV のとき。 |

<details><summary>低レベル</summary>

`AvatarEventName`、`AvatarEvents`、`AvatarHandle`、`AvatarListing`、`AvatarPackError`、`AvatarSet`、`CpuSentenceModel`、`CreateOptions`、`Emitter`、`FRAMES_PACK_MAGIC`、`FrameKernel`、`FramePrecision`、`FrameTiming`、`FramesPackIndex`、`Geometry`、`IDLE_GOOD_PAIR_DIST`、`IdleCache`、`IdlePairChoice`、`IdlePairTable`、`IdleSwitchFile`、`LipAvatar`、`LipModels`、`LipWeights`、`LoadProgress`、`LogMel`、`MEL_SAMPLE_RATE`、`MODEL_FILES`、`NodeBackend`、`NotImplementedError`、`OptionsError`、`PackData`、`PackMeta`、`PackRef`、`PackSource`、`RenderAudioOptions`、`RenderAudioProgress`、`ResolvedOptions`、`SentenceInput`、`SentenceWeights`、`SpeechAudio`、`SpeechPrep`、`ToneTables`、`VERSION`、`VideoContainer`、`WavInfo`、`WebGPUUnavailableError`、`checkAvatarSet`、`checkPackMeta`、`chooseIdlePair`、`fileHash`、`framesPackFrameEnd`、`framesPackHeadLength`、`inferFrames`、`joinAssets`、`listAvatar`、`loadPackData`、`logMel`、`modelPath`、`nativeAddonPath`、`nodePackSource`、`parseSentenceWeights`、`readFramesPackIndex`、`resample`、`resolveOptions`、`sentenceLatents`、`sha256Hex`、`speechMask`、`wavAudio`

</details>

## ZEN-TTS で話させる（ブラウザ）

`@zenryokukikai/zen-lipsync/tts`

| 名前 | 形 | 説明 |
|---|---|---|
| `speakText` | <code>speakText(avatar: AvatarHandle, tts: TtsEngine, text: string, options?: SpeakTextOptions): Promise&lt;void&gt;</code> | 文章を ZEN-TTS で話させます。 |
| `SpeakTextOptions` | <code>TtsSpeakOptions + { onSentence? }</code> | speakText() のオプション。 |
| `TtsEngine` | <code>{ speak() }</code> | ZenTts／ZenTtsNode の形。 |
| `splitSentences` | <code>splitSentences(text: string, maxChars?: number): string[]</code> | 文章を文に分けます。 |
| `TTS_SAMPLE_RATE` | <code>const TTS_SAMPLE_RATE = 44100;</code> | 44100。 |

<details><summary>低レベル</summary>

`TtsSpeakOptions`、`ttsChunks`

</details>

## ZEN-TTS で話させる（Node.js）

`@zenryokukikai/zen-lipsync/tts/node`

| 名前 | 形 | 説明 |
|---|---|---|
| `renderText` | <code>renderText(avatar: NodeAvatar, tts: TtsEngine, text: string, opts: RenderOptions & TextOutputOptions)</code> | 文章 → 動画ファイル。 |
| `streamText` | <code>streamText(avatar: NodeAvatar, tts: TtsEngine, text: string, opts: StreamOptions & TextOutputOptions)</code> | 文章 → 配信。 |
| `speakTextFrames` | <code>speakTextFrames(avatar: NodeAvatar, tts: TtsEngine, text: string, opts?: FramesOptions & TextOutputOptions)</code> | 文章 → フレーム。 |
| `TextOutputOptions` | <code>{ tts? }</code> | tts オプション。 |

<details><summary>低レベル</summary>

`SpeakTextOptions`、`TTS_SAMPLE_RATE`、`TtsEngine`、`TtsSpeakOptions`、`splitSentences`、`ttsChunks`

</details>

## 会話（マイク → 返事 → 声 → アバター）

`@zenryokukikai/zen-lipsync/conversation`

| 名前 | 形 | 説明 |
|---|---|---|
| `Conversation` | <code>{ turnState, messages, on(), start(), stop(), destroy(), userSpeechStart(), speechEnd(), userFinal(), sendText(), stats(), turnLatencies() }</code> | 会話の進行役。 |
| `ConversationOptions` | <code>{ avatar, reply(), synthesize(), aizuchi?, emotion?(), idleAngry? }</code> | Conversation のオプション。 |
| `ReplyContext` | <code>{ messages, utt, signal }</code> | reply() の引数。 |
| `SynthesisContext` | <code>{ piece, text, index, utt, signal, kind }</code> | synthesize() の引数。 |
| `ConversationMessage` | <code>{ role, content }</code> | 1 発言。 |
| `ConversationEvents` | <code>{ speechstart, speechend, replystart, replysound, replydone, turnend, error, latency }</code> | on() のイベント。 |
| `ConversationStats` | <code>{ turns, medians, bargeIns, maxBargeInStopMs, errors }</code> | stats() の戻り値。 |
| `TurnLatency` | <code>{ utt, speechEndToFinalMs, speechEndToTokenMs, speechEndToSoundMs, speechEndToReplyDoneMs, replyDoneToPlaybackEndMs, bargeIn, aizuchi, aborted }</code> | 1 ターンごとの遅れ。 |
| `EmotionReading` | <code>{ angry }</code> | emotion() の戻り値。 |
| `openMic` | <code>openMic(o: MicOptions): Promise&lt;Mic&gt;</code> | マイクの話し始め・話し終わりを伝えます。 |
| `MicOptions` | <code>EndpointOptions + { conversation, isSpeech?(), onFrame?(), context?, sampleRate? }</code> | openMic() のオプション。 |
| `Mic` | <code>{ stop(), end(), inSpeech, levelDbfs }</code> | openMic() の戻り値。 |

<details><summary>低レベル</summary>

`EndPointer`、`EndpointEvent`、`EndpointOptions`、`PushQueue`、`TurnAction`、`TurnData`、`TurnEvent`、`TurnState`、`TurnStep`、`initialTurnData`、`turnStep`

</details>
