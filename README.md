# ZEN-LipSync SDK（@zenryokukikai/zen-lipsync）

実写アバターの音声駆動リップシンク SDK。ブラウザ（WebGPU）と Node.js（CPU）で動作します。

バージョン 1.1.0。

各アバターの voice（ZEN-TTS）を Node.js（CPU）でレンダリングしたサンプルです。

<table>
<tr>
<th>全力 マリン</th>
<th>全力 ハシロウ</th>
<th>全力 キッカ</th>
</tr>
<tr>
<td><video controls src="https://github.com/user-attachments/assets/7f2fbd5a-8f02-4261-b935-5bfe649ffc43" width="260"></video></td>
<td><video controls src="https://github.com/user-attachments/assets/e288e919-1e29-4c58-83f1-aa494fc2cb77" width="260"></video></td>
<td><video controls src="https://github.com/user-attachments/assets/7148d0ad-7327-4ead-9e12-8b7cc3b39213" width="260"></video></td>
</tr>
</table>

| アバター | id | model | サイズ |
|---|---|---|---|
| 全力 マリン（ぜんりょく まりん） | `marine` | marine-1.0 | 84 MB |
| 全力 ハシロウ（ぜんりょく はしろう） | `hashiro` | hashiro-1.0 | 95 MB |
| 全力 キッカ（ぜんりょく きっか） | `kikka` | kikka-1.0 | 56 MB |

アバターパック（モデルと素材）は [zenryokukikai/zen-lipsync-avatars](https://github.com/zenryokukikai/zen-lipsync-avatars )。モデルはアバターごとに学習済みで、他の顔・映像には使えません。新しいアバターのモデル作成は全力機械（https://zenryokukikai.com ）へご依頼ください。

利用条件として、アプリ・サイト・作品の利用者が見つけられる場所に、クレジット「リップシンク：ZEN-LipSync（https://zenryokukikai.com ）」を表示してください。URL はリンクにします（[LICENSE.md](LICENSE.md) 3 章）。

## 関連リポジトリ

| リポジトリ | 中身 |
|---|---|
| [zenryokukikai/zen-lipsync](https://github.com/zenryokukikai/zen-lipsync)（このリポジトリ） | SDK（`@zenryokukikai/zen-lipsync`）：ブラウザ・React・Node・ZEN-TTS 連携 |
| [zenryokukikai/zen-tts](https://github.com/zenryokukikai/zen-tts) | ZEN-TTS：4 voices の重み、ブラウザ / Node ランタイム |
| [zenryokukikai/zen-lipsync-avatars](https://github.com/zenryokukikai/zen-lipsync-avatars) | アバターパック（marine / hashiro / kikka）と examples |

## 入れ方

npm レジストリには公開していません。GitHub から指定します。

```sh
# npm
npm i github:zenryokukikai/zen-lipsync#v1.1.0
# pnpm
pnpm add github:zenryokukikai/zen-lipsync#v1.1.0
# yarn
yarn add github:zenryokukikai/zen-lipsync#v1.1.0
```

- `#v1.1.0` はタグまたはコミットハッシュで固定してください。
- Node.js 26.12.1（`.mise.toml` で固定、`mise install`）。
- アバターパックは同梱されません。zen-lipsync-avatars を配置して `avatar` / `assetsPath` / `assetsBaseUrl` で指定します。
- TTS 連携には `github:zenryokukikai/zen-tts#v1.0.0` を追加します。
- 開発：`pnpm install && pnpm build && pnpm test`。`pnpm-workspace.yaml` の `minimumReleaseAge` で公開 7 日未満の依存を除外。

## エントリポイント

| entry | import | 内容 |
|---|---|---|
| ブラウザ | `@zenryokukikai/zen-lipsync` | WebGPU のみ。log-mel、文モデル、フレームモデル、合成まで端末内で実行（音声・映像を送信しない）。WebGPU 非対応時は `WebGPUUnavailableError`。 |
| React | `@zenryokukikai/zen-lipsync/react` | `<ZenLipSync>` コンポーネント。 |
| Node.js | `@zenryokukikai/zen-lipsync/node` | Node.js 26.12.1+、CPU。RGB フレーム、MP4 / WebM ファイル、配信ストリーム（fMP4 / WebM / JPEG）。 |
| TTS 連携 | `@zenryokukikai/zen-lipsync/tts`、`…/tts/node` | ZEN-TTS で文単位に合成しながら、ギャップなくリップシンク。 |
| 対話 | `@zenryokukikai/zen-lipsync/conversation` | マイクのターン検出 → reply（LLM）→ synthesize（TTS）→ アバター。barge-in、相づち、感情による idle 切り替え、レイテンシ計測。LLM / TTS はアプリが渡す。 |

`VERSION`（`'1.1.0'`）をブラウザ / Node の両エントリから export。パック形式は [docs/AVATAR-PACK.md](docs/AVATAR-PACK.md)。

## 使い方（ブラウザ）

`container` の中に枠・待機映像・リップシンクを作り、幅いっぱい・映像と同じ縦横比で合わせます（`fit: 'contain'`（既定）で枠に収める、`'cover'` で枠いっぱいにして切る）。画面の向き・大きさの変更や devicePixelRatio 1〜3 でも顔からずれません。

```ts
import { create, WebGPUUnavailableError } from '@zenryokukikai/zen-lipsync';

try {
  const avatar = await create({
    container: document.getElementById('avatar')!,                // SDK がこの中に枠・映像・リップシンクを作ります
    avatar: 'https://example.com/avatars/marine/avatar.json',     // zen-lipsync-avatars の avatars/marine/
    onProgress: ({ loadedBytes }) => console.log(loadedBytes),
  });
  avatar.on('speechstart', ({ frames }) => console.log('話し始め', frames));
  avatar.on('speechend', ({ interrupted }) => console.log('話し終わり', interrupted));
  await avatar.speak({ pcm, sampleRate: 44100 });       // モノラル Float32Array、任意のサンプリングレート
  await avatar.speakStream(chunks);                     // 少しずつ届く音声（AsyncIterable<{ pcm, sampleRate }>）
  await avatar.setIdle('angry');
  avatar.stop(); avatar.pause(); avatar.resume();
  console.log(avatar.frameTiming());                    // { frames, inferMsPerFrame }
  avatar.destroy();
} catch (e) {
  if (e instanceof WebGPUUnavailableError) { /* WebGPU が使えない */ }
}
```

React：`<ZenLipSync avatar="/avatars/marine/avatar.json" onReady={(avatar) => avatar.speak({ pcm, sampleRate })} onError={console.error} />`（`@zenryokukikai/zen-lipsync/react`）。

| API | 説明 |
|---|---|
| `create({ container, avatar, fit?, weights?, idle?, onProgress? })` | アバターを読み込み、待機映像を流し始めます。`fit`：`'contain'`（既定）・`'cover'`。`weights`：`'i8'`（既定）・`'f32'`・`'i8q'`。 |
| `speak(audio)` / `speakStream(chunks)` | 話させます。鳴り終わると解決します。`speakStream` は届いた順につなげて話します。 |
| `setIdle(name)` | 待機映像を切り替えます（話していない間だけ。似たフレームで 0.3 秒のクロスフェード）。 |
| `on(event, listener)` | `ready`・`speechstart`・`speechend`・`idlechange`・`error`。戻り値の関数で解除します。 |
| `stop()` / `pause()` / `resume()` / `destroy()` | 止める（待機映像は続く）・一時停止・再開・解放。 |
| エラー | `WebGPUUnavailableError`（`reason`：`'no-navigator-gpu'`・`'no-adapter'`）、`OptionsError`、`AvatarPackError` |

**口元と待機映像のコマの合わせ方**：口元は、画面に出ている待機映像のコマに合わせて、その上の canvas に描きます。ブラウザによっては canvas が待機映像より 1 回遅れて画面に出るため（iPhone の Safari など）、既定（`sync: 'auto'`）では端末ごとに描画が間に合っているかを測り、いつも遅れる端末では 1 コマ先の口元を描きます。測れない端末（フレームの表示時刻が別の時計のもの）は、待機中の最初の 2 コマで 1 コマ先に切り替えます。`LipSyncPlayer.create(…, { sync })` で `'video'`（そのコマ）・`'next'`（1 コマ先）・`'canvas'`（待機映像のコマと口元を同じ canvas に描く）に固定できます。

**待機映像の切り替え（`setIdle`）**：待機ループがいくつあっても、`<video>` は 2 つ（表示中と待機用）だけ使います。待機用の `<video>` に次のループを読み込み、ペアのフレームへシークして再生し、実際にコマが出てからクロスフェードして、2 つの役割を入れ替えます。iOS の Safari（低電力モードなど）はタップの外での `play()` を断ることがあるため、`resume()` はユーザーのタップの中で呼んでください（2 つの `<video>` をそこで一度再生します）。読み込み・シーク・再生のどれかが時間内に終わらないときは、切り替えをやめて今のループのまま続けます（`setIdle` は `false`）。表示中の `<video>` からコマが 0.7 秒届かないときは、再生し直します。

動くブラウザ：WebGPU が使えるもの（Chrome / Edge 113 以降、Safari 26 以降、Android の Chrome など）。細かく使う場合：`LipSyncPlayer`、`Avatar`、`loadModels`、`logMel`、`sentenceLatents`、`inferFrames`、`Compositor`（`src/index.ts`）。

## 使い方（サーバー・Node.js）

Node.js 26.12.1+、CPU のみ（GPU 不使用）。

**必要なもの**
- **ffmpeg**：待機映像の読み込みと動画の書き出しに使います。`PATH` のものを使います（`apt install ffmpeg` など）。別の場所なら `ffmpegPath` か環境変数 `ZEN_LIPSYNC_FFMPEG`。
- **ネイティブ版（任意、おすすめ）**：linux-x64 なら `pnpm run build:native`（`native/node/build.sh`）でネイティブ版を作れます（gcc が必要）。既定の float32（ブラウザと同じ画質）はネイティブ版で動きます。作らない場合と、ほかの OS は WebAssembly 版の int8 で動きます。
- アバターのファイル（zen-lipsync-avatars を clone したディレクトリ、または配信している URL）。

```js
import { create } from '@zenryokukikai/zen-lipsync/node';

const avatar = await create({ avatar: 'marine', assetsPath: '/path/to/zen-lipsync-avatars' });
await avatar.render({ pcm, sampleRate: 24000 }, { out: 'marine.mp4' });   // MP4（H.264 + AAC）。.webm なら VP9 + Opus
for await (const f of avatar.frames({ pcm, sampleRate })) { /* f.index, f.timeMs, f.width, f.height, f.rgb, f.audio */ }
const stream = avatar.stream(speechChunks, { format: 'jpeg', realtime: true, width: 640 });   // 'fmp4'・'webm'・'jpeg'
await avatar.setIdle('angry');   // 0.3 秒のクロスフェード
avatar.destroy();
```

- **音声を少しずつ渡す**：`frames()`／`render()`／`stream()` には、音声 1 つの代わりに、音声のチャンクの（非同期）イテレータも渡せます。次のチャンクの準備は別スレッドで、今のチャンクを描いている間に進みます。音声は隙間なく繋がります。
- **`realtime: true`**：1 秒 25 フレームずつ。次の音声が届くまでは待機映像を出し続けます。出力は `startDelayMs`（既定 600 ms）遅れて始まります。
- **`create()` のオプション**：`backend`（`'auto'`（既定：ネイティブ版があればそれ、なければ WebAssembly）、`'cpu-native'`、`'cpu-wasm'`）、`idle`、`assetsBaseUrl`（`assetsPath` の代わりに URL）、`idleCache`（`'stream'`（既定）／`'memory'`）、`precision`（`'float32'`（ブラウザと同じ画質。ネイティブ版だけ）／`'int8'`（速い））。指定しないときは、ネイティブの float32 版が読み込めれば float32、読み込めなければ（`backend: 'cpu-wasm'`、ネイティブ版がない、AVX2 がない CPU）int8 です。どの版で動いているかは `avatar.kernel`（float32：`avx512-f32`／`avx2-f32`、int8：`avx512`／`avx2`／`sse41`／`wasm`）。
- 例：zen-lipsync-avatars の `examples/node-mp4/`（WAV → MP4）、`examples/node-stream/`（WebSocket で配信）、`examples/node-tts/`（文章 → MP4）、`examples/conversation/`（ブラウザで会話）。

### 音声ファイルから長い動画を作る

WAV → 動画ファイル。30 秒〜数分の音声を想定し、チャンク単位で読み込むためメモリは音声長に比例しません。

```js
import { renderAudioFile } from '@zenryokukikai/zen-lipsync/node';

const r = await renderAudioFile({
  avatar: 'marine', assetsPath: '/path/to/zen-lipsync-avatars',
  audio: 'lesson.wav', out: 'lesson.mp4',
  crf: 20, idleBeforeMs: 500, idleAfterMs: 500,
  onProgress: (p) => console.log(`${(p.timeMs / 1000).toFixed(0)} 秒`, p.ratio),   // 約 1 秒ごと
});
// r: { file, frames, durationMs, audioMs, wallMs, fps, rtf }
```

| オプション | 内容 |
|---|---|
| `avatar` / `assetsPath` | アバター（`'marine'`・`'hashiro'`・`'kikka'`）と、そのファイルの場所。`create()` と同じ |
| `audio` / `out` | 入力の WAV、出力の動画（`.mp4`：H.264 + AAC、`.webm`：VP9 + Opus） |
| `crf` | 画質（小さいほど高画質。既定 20（H.264）／32（VP9）） |
| `idleBeforeMs` / `idleAfterMs` | 話し始める前と後の待機映像（ms、既定 0） |
| `chunkSeconds` | 一度に渡す音声の長さ（秒、既定 8）。小さくするとメモリが減り、大きくすると少し速い |
| `onProgress` | `{ frames, timeMs, totalFrames, ratio, fps, wallMs }` |
| `signal` | `AbortSignal`。中断すると途中の動画ファイルは消えます |

- **WAV だけ**：8/16/24/32 bit の PCM と 32/64 bit の float、モノラル・ステレオ（平均してモノラルにします）。ほかの形式は先に変換してください：`ffmpeg -i in.mp3 -ac 1 -ar 48000 out.wav`。
- **メモリ**：プロセス全体で約 0.4 GB（`idleCache: 'stream'`、実測 354〜425 MB）。30 秒でも 3 分でもほぼ同じで、音声の長さに比例して増えません。
- **速さ**（Intel Xeon Silver 4316、CPU 1 スレッド、ネイティブ AVX-512 VNNI の int8、33 秒と 180 秒の音声）：1 秒あたり約 50〜59 フレーム、実時間の 0.42〜0.50 倍。180 秒の動画が約 78 秒でできます。既定の float32 と、AVX2／WebAssembly 版は下の「速さ」の表のとおり遅くなります。
- 同じアバターで何本も作るときは、`create()` と `renderAudio(avatar, chunks, opts)` を分けて使います（アバターの読み込みが 1 回で済みます）。
- 例：`node examples/render-audio.mjs marine /path/to/zen-lipsync-avatars in.wav out.mp4`

**速さ**（Intel Xeon Silver 4316、CPU 1 本に固定、8.12 秒の音声、動画の書き出しを除く）

| 版 | フレームモデル（1 フレーム） | 全体（合成込み） | 実時間に対する時間 |
|---|---|---|---|
| ネイティブ AVX-512、float32（既定） | 9.0〜10.0 ms | 約 26〜27 fps | 0.93〜0.95 |
| ネイティブ AVX2、float32（既定） | 15.4〜16.2 ms | 約 23 fps | 1.08 |
| ネイティブ AVX-512 VNNI、`precision: 'int8'` | 2.9 ms | 約 40 fps（`idleCache: 'memory'` なら約 47 fps） | 0.62（0.52） |
| ネイティブ AVX2、`precision: 'int8'` | 26.6 ms | 約 20 fps | 1.22 |
| ネイティブ SSE4.1、`precision: 'int8'` | 29.8 ms | 約 19 fps | 1.30 |
| WebAssembly（SIMD128）、int8 | 29.0 ms | 約 19.5 fps | 1.27 |

float32（既定）は、フレームモデルの計算をすべて float32 で行います（重みは `frame.i8.wgw` を読み込み時に一度だけ float32 に戻したもので、ブラウザと同じ出力になります）。`precision: 'int8'` は途中の計算も int8 に丸めるため速く、アバターによっては口元のまわりに細かい粒が出ます。float32 の 2 行は、ほかの処理が動いているマシン（ロードアベレージ 4〜7）で 10.43 秒の音声を測ったもので、同じ条件の int8（AVX-512 VNNI）は約 32〜33 fps、0.76〜0.78 でした。

1 未満なら実時間より速く作れます。CPU 1 本でリアルタイムに間に合うのは AVX-512 の float32 と AVX-512 VNNI の int8 です（AVX2／SSE4.1／WebAssembly 版で `realtime: true` の配信をするなら、AVX-512 の CPU か、文モデル用にもう 1 本おすすめします）。音声 1 秒あたり、文モデルなどの準備に CPU 約 0.3 秒を別スレッドで使います。

**同時に動かす数とメモリ**
- 1 つのアバターは、同時に 1 つの出力だけを作ります。並行して作るときは `create()` をその数だけ呼びます。モデルの重みはプロセスの中で共有します。
- アバター 1 つあたりのメモリ：`idleCache: 'stream'`（既定）で約 110 MB、`'memory'` で約 4.4 GB（待機映像の全フレームを持ち、少し速くなります）。
- ネイティブ版の int8 のフレームモデルは 1 プロセスに 1 つです。AVX-512 VNNI 版はスレッドごとに作業領域を持ち、AVX2／SSE4.1 版は 1 プロセスの中で順番に実行します。CPU を何本も使うときは、プロセスを分けるのがおすすめです。
- `precision: 'float32'` のフレームモデルは、1 プロセスにいくつでも作れて、どのスレッドからも同時に実行できます（作業領域はスレッドごとに約 11 MB）。

## ZEN-TTS で文章を話させる

ZEN-TTS（`zen-tts`）は、組み合わせたいときだけ入れる追加のパッケージです（`pnpm add github:zenryokukikai/zen-tts`）。

```js
// ブラウザ（WebGPU）
import { create } from '@zenryokukikai/zen-lipsync';
import { speakText } from '@zenryokukikai/zen-lipsync/tts';
import { ZenTts } from 'zen-tts';
const avatar = await create({ container, avatar: '/avatars/marine/avatar.json' });
const tts = await ZenTts.load({ modelsUrl: './tts-models', backend: 'webgpu' });
await speakText(avatar, tts, 'こんにちは。全力マリンです。今日はいい天気ですね。', { emotion: { happy: 60, neutral: 40 } });

// サーバー（Node.js、CPU）
import { renderText, streamText } from '@zenryokukikai/zen-lipsync/tts/node';
import { ZenTtsNode } from 'zen-tts/node';
const [av, nodeTts] = await Promise.all([create({ avatar: 'marine', assetsPath }), ZenTtsNode.load({ modelsPath: './tts-models' })]);
await renderText(av, nodeTts, 'こんにちは。全力マリンです。', { out: 'marine.mp4' });              // 動画ファイル
const live = streamText(av, nodeTts, '配信のテストです。', { format: 'jpeg', realtime: true });   // 配信
```

- 文章は「。！？」と改行で区切り、1 文ずつ ZEN-TTS で作ります。できた文から順に渡すので、最初の文が流れている間に次の文を作ります。文と文の間にすき間はできません。
- `speakText`／`renderText`／`streamText` の最後の引数には、ZEN-TTS の `speak()` のオプション（`emotion`、`change`、`strength`、`dict` など）と `onSentence`（1 文できるたびに呼ばれる）をそのまま入れられます。Node 版は `tts: { ... }` に入れ、ほかは `frames()`／`render()`／`stream()` のオプションです。
- 声も出力の音も 44.1 kHz です。使うときは、ZEN-TTS と ZEN-LipSync の両方のクレジットを表示してください（必須）。1 行にまとめて構いません。例：「音声合成：ZEN-TTS / リップシンク：ZEN-LipSync（https://zenryokukikai.com ）」（URL はリンクにします）

## 会話（マイク → 返事 → 声 → アバター）

話しかけると返事をする形は、`conversation` の入口で作れます。返事の文章（LLM）と声（TTS）はアプリが渡し、SDK は「いつ話し終わりか」「いつ音を止めるか」「待機を切り替えるか」「どこで何 ms かかったか」を持ちます。

```js
import { create } from '@zenryokukikai/zen-lipsync';
import { Conversation, openMic } from '@zenryokukikai/zen-lipsync/conversation';

const avatar = await create({ container, avatar: '/avatars/kikka/avatar.json' });

const conversation = new Conversation({
  avatar,
  // 返事：文章を少しずつ返す（LLM でも台本でもよい）
  async *reply({ messages, signal }) {
    for await (const piece of myLlm(messages, signal)) yield piece;
  },
  // 声：渡された文を、できた順に音にする（ZEN-TTS は speakStream がそのまま使えます）
  async *synthesize({ piece, signal }) {
    for await (const c of tts.speakStream(piece)) yield { pcm: c.audio, sampleRate: c.sampleRate ?? 44100 };
  },
  // 任意：話し終わりの相づち
  aizuchi: { pick: ({ messages }) => pickAizuchi(messages) },
  // 任意：感情（怒りの強さ）で待機ループを切り替える。怒りの待機があるアバターだけ動きます
  emotion: (text) => ({ angry: emotionOf(text) }),
});

conversation.start();
const mic = await openMic({ conversation });   // 話し始めで音を止め、静かになったらターン終了
// マイクを使わないとき（テスト・キーボード）：conversation.sendText('こんにちは')
```

- **入力の渡し方**：会話は、ユーザーの発話を次の 3 つのイベントで受け取ります。
  | メソッド | 呼ぶタイミング | 動作 |
  |---|---|---|
  | `userSpeechStart()` | ユーザーが話し始めたとき（VAD の発話開始） | アバターが話していれば止める（barge-in） |
  | `speechEnd()` | ユーザーが話し終えたとき（VAD の発話終了） | ターンを終え、相づちを流す。遅延計測の起点 |
  | `userFinal(text)` | 音声認識の確定結果が出たとき | `reply()` を呼んで返事を作る |
  - `openMic()` を使う場合、`userSpeechStart()` と `speechEnd()` は自動で呼ばれます。音声認識は含まないので、認識結果を `userFinal(text)` で渡します。
  - 自前のマイク・VAD・音声認識を使う場合は、それぞれのタイミングで 3 つを呼びます。
  - `sendText(text)` は 3 つを続けて呼ぶ省略形です。テキスト入力やテストに使います。
- **割り込み（barge-in）**：`userSpeechStart()` で音を止め、まだ作っている途中の返事も捨てます。止まった時間は `stats().maxBargeInStopMs` で見られます。
- **遅れの計測**：`stats()` が中央値を返します（話し終わり→確定、→最初の文字、→最初の音）。1 ターンごとの値は `turnLatencies()`、`on('latency', …)` でも取れます。
- **相づち**：`aizuchi.pick()` が文を返すと、話し終わりにそれを先に流し、返事はそのあとに続きます。
- **待機の切り替え**：`emotion()` が返す怒りの強さが 0.5 以上で `angry`、0.4 以下で `normal`（`idleAngry` で変更可）。怒りの待機が無いアバターでは何もしません。
- マイクは 16 kHz・32 ms のフレームで、静かさが 700 ms 続くとターン終了（`openMic` の `endSilenceMs` などで変更）。エコーキャンセルは有効にします（アバターの声を拾わないため）。VAD モデルや話し終わり判定のモデルを使う場合は `isSpeech` に渡すか、`end()` を自分で呼びます。

例：`examples/conversation/`（台本の reply ＋ ZEN-TTS。avatar・voice・TTS backend をリストで選択し、latency とログを表示。ZEN-TTS を読み込めない場合はエラーで停止）。`pnpm build` 後、リポジトリのルートを配信して開きます（マイクと WebGPU のため https か localhost）。アバターは `?assets=<zen-lipsync-avatars のパス>/`、ZEN-TTS は `?tts=<zen-tts の dist/index.js>`（モデルは `?models=`）。

会話キット自体のレイテンシ上乗せは 1〜2 ms（reply の最初のトークン → 最初の音 約 20 ms、ZEN-TTS ブラウザ CPU）。barge-in は発音中から 0.3 ms で停止します。

## 速さ

| 環境 | 1 フレームの推論時間 |
|---|---|
| Intel Xeon Silver 4316、CPU 1 スレッド、ネイティブ（AVX-512 VNNI、`native/x86/`） | 2.85 ms（音声特徴を含む。フレームモデルだけなら 2.65 ms） |
| iPhone Air、Safari、WebGPU | 約 146 fps |
| iPhone 16 Pro、Safari、WebGPU | 約 131 fps |
| Pixel 9、Chrome、WebGPU | 約 61 fps |
| MacBook Pro（M3 Max）、ブラウザ、WebGPU | 約 1,060 fps |

ネイティブの CPU 版は `native/x86/build.sh` でビルドし、`native/x86/bench.sh <アバターの models ディレクトリ>` で測れます（AVX-512 VNNI が使える x86-64 CPU）。WebAssembly 版（`cpu/`）は `scripts/build-cpu-wasm.sh`（Emscripten）で作ります。Node のネイティブ版（AVX-512 VNNI／AVX2／SSE4.1 を自動で選ぶ）は `native/node/build.sh` です。

## ライセンス

**ZEN-LipSync ライセンス**（[LICENSE.md](LICENSE.md)）。SDK とアバター（zen-lipsync-avatars）は、同じ文面のこのライセンスです。

- 本ソフトウェアをそのまま自分のアプリ・サイト・サービス・作品に組み込んで使えます（商用・非商用を問わず）。改変はできません（ソースコード・モデルの重み・アバター素材を含む。自分のアプリのビルドで依存として取り込み、バンドル・圧縮することは改変にあたりません）。
- **条件**：本ソフトウェアやそれで作った映像を使うアプリ・サイト・作品では、利用者が見つけられる場所にクレジット「リップシンク：ZEN-LipSync（https://zenryokukikai.com ）」を表示してください。URL はリンクにします（リンクを埋め込めない媒体では URL を文字で書けば足ります）。
- 再配布（モデル・重み・アバター素材を第三者に渡す、再許諾、販売、譲渡、貸与）はできません。他の機械学習モデルの学習・蒸留・評価への利用もできません。なりすまし、本人の同意のない利用、人をだます目的などにも使えません（3 章）。
- アバター（全力 マリン・全力 ハシロウ・全力 キッカ）の素材の決まりは 4 章です（素材は zen-lipsync-avatars にあります）。第三者の部分は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
