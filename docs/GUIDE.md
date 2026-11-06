# ZEN-LipSync 組み込みガイド

音声に合わせて、実写のアバターの口を動かします。例はそのまま動きます。関数の一覧は [API.md](API.md)。

## 入れ方

```sh
npm i github:zenryokukikai/zen-lipsync#v1.1.0 github:zenryokukikai/zen-lipsync-avatars#v1.1.0
npm i github:zenryokukikai/zen-tts#v1.2.0     # 文章を話させるとき
```

- ブラウザは WebGPU が必要です（ないと `WebGPUUnavailableError`）。Node.js は 26.12.1 以降と ffmpeg。
- **クレジットの表示が必須**：「リップシンク：ZEN-LipSync（https://zenryokukikai.com ）」（URL はリンク）。ZEN-TTS も使うときは「音声合成：ZEN-TTS / リップシンク：ZEN-LipSync（https://zenryokukikai.com ）」。

## どれを使うか

| やりたいこと | import | 例 |
|---|---|---|
| Web ページで話させる | `@zenryokukikai/zen-lipsync` | [ブラウザ](#ブラウザ) |
| React に入れる | `@zenryokukikai/zen-lipsync/react` | [React](#react) |
| 音声ファイル → MP4 | `@zenryokukikai/zen-lipsync/node` | [MP4](#mp4nodejs) |
| サーバーで作りながら配信 | `@zenryokukikai/zen-lipsync/node` | [配信](#配信nodejs) |
| 文章を話させる（ZEN-TTS） | `…/tts`、`…/tts/node` | [文章](#文章を話させるzen-tts) |
| 話しかけると答える | `@zenryokukikai/zen-lipsync/conversation` | [会話](#会話) |

## アバター

| アバター | ブラウザ（`avatar`） | Node.js（`avatar`） | 待機映像 |
|---|---|---|---|
| 全力 マリン | `avatars/marine/avatar.json` | `'marine'` | `normal`・`angry` |
| 全力 ハシロウ | `avatars/hashiro/avatar.json` | `'hashiro'` | `normal`・`angry` |
| 全力 キッカ | `avatars/kikka/avatar.json` | `'kikka'` | `normal` |

ブラウザは `assetsBaseUrl`（zen-lipsync-avatars を置いた URL）、Node.js は `assetsPath`（そのフォルダ）と組み合わせます。

## ブラウザ

`index.html` をプロジェクトのフォルダに置き、そのフォルダを配信して開きます（例：`npx serve .`）。

```html
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>ZEN-LipSync</title>
  <script type="importmap">
    { "imports": { "@zenryokukikai/zen-lipsync": "/node_modules/@zenryokukikai/zen-lipsync/dist/index.js" } }
  </script>
</head>
<body>
  <div id="avatar" style="max-width: 640px"></div>
  <button id="talk">話させる</button>
  <p>リップシンク：ZEN-LipSync（<a href="https://zenryokukikai.com">https://zenryokukikai.com</a>）</p>

  <script type="module">
    import { create, WebGPUUnavailableError } from '@zenryokukikai/zen-lipsync';

    // zen-lipsync-avatars のファイルを置いた場所（ここでは node_modules の中）
    const assetsBaseUrl = '/node_modules/zen-lipsync-avatars/';

    document.getElementById('talk').onclick = async () => {
      try {
        const avatar = await create({
          container: document.getElementById('avatar'),
          assetsBaseUrl,
          avatar: 'avatars/marine/avatar.json',
        });
        // 音声（WAV など）を読み、モノラルの Float32Array にして渡す
        const wav = await (await fetch(assetsBaseUrl + 'avatars/marine/normal/sample.wav')).arrayBuffer();
        const audio = await new AudioContext().decodeAudioData(wav);
        await avatar.speak({ pcm: audio.getChannelData(0), sampleRate: audio.sampleRate });
      } catch (e) {
        alert(e instanceof WebGPUUnavailableError ? 'このブラウザは WebGPU が使えません' : e);
      }
    };
  </script>
</body>
</html>
```

- `speak()` にはモノラルの `Float32Array` とサンプリングレート（何 Hz でも可）を渡します。
- 少しずつ届く音声は `speakStream()`、待機映像の切り替えは `setIdle('angry')`。

## React

```jsx
import { useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ZenLipSync } from '@zenryokukikai/zen-lipsync/react';

const assetsBaseUrl = '/node_modules/zen-lipsync-avatars/';

function App() {
  const avatarRef = useRef(null);
  const [ready, setReady] = useState(false);
  const talk = async () => {
    const wav = await (await fetch(assetsBaseUrl + 'avatars/marine/normal/sample.wav')).arrayBuffer();
    const audio = await new AudioContext().decodeAudioData(wav);
    await avatarRef.current.speak({ pcm: audio.getChannelData(0), sampleRate: audio.sampleRate });
  };
  return (
    <>
      <ZenLipSync
        assetsBaseUrl={assetsBaseUrl}
        avatar="avatars/marine/avatar.json"
        onReady={(avatar) => { avatarRef.current = avatar; setReady(true); }}
        onError={(e) => alert(e.message)}
        style={{ maxWidth: 640 }}
      />
      <button onClick={talk} disabled={!ready}>話させる</button>
      <p>リップシンク：ZEN-LipSync（<a href="https://zenryokukikai.com">https://zenryokukikai.com</a>）</p>
    </>
  );
}

createRoot(document.getElementById('root')).render(<App />);
```

- `onReady` に、ブラウザの例の `avatar` と同じものが渡されます。

## MP4（Node.js）

`node mp4.mjs` で、話す動画（1264×720、25 fps）ができます。

```js
// WAV → MP4（全力 マリンが話す動画）
import { renderAudioFile } from '@zenryokukikai/zen-lipsync/node';

const r = await renderAudioFile({
  avatar: 'marine',
  assetsPath: 'node_modules/zen-lipsync-avatars',          // zen-lipsync-avatars の場所
  audio: 'node_modules/zen-lipsync-avatars/avatars/marine/normal/sample.wav',
  out: 'marine.mp4',
});
console.log(`${r.file}：${r.frames} フレーム、${(r.durationMs / 1000).toFixed(2)} 秒`);
```

## 配信（Node.js）

`node stream.mjs` → http://localhost:8091/ で、作りながら実時間で流れます。

```js
// 話す映像を、作りながら配信する（WebM：VP9 + Opus）。node stream.mjs → http://localhost:8091/
import { create, openWav } from '@zenryokukikai/zen-lipsync/node';
import { createServer } from 'node:http';

const assetsPath = 'node_modules/zen-lipsync-avatars';
const page = `<!doctype html><meta charset="utf-8">
<video src="/talk.webm" controls autoplay playsinline width="640"></video>
<p>リップシンク：ZEN-LipSync（<a href="https://zenryokukikai.com">https://zenryokukikai.com</a>）</p>`;

createServer(async (req, res) => {
  if (req.url !== '/talk.webm') { res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' }); res.end(page); return; }
  const avatar = await create({ avatar: 'marine', assetsPath });
  // 音声を 1 秒ずつ渡す（音声合成が少しずつ返す音声でも、同じように渡せます）
  const wav = await openWav(`${assetsPath}/avatars/marine/normal/sample.wav`);
  async function* speech() {
    for (let at = 0; at < wav.frames; at += wav.sampleRate) yield { pcm: await wav.read(at, wav.sampleRate), sampleRate: wav.sampleRate };
    await wav.close();
  }
  res.writeHead(200, { 'content-type': 'video/webm' });
  const stream = avatar.stream(speech(), { format: 'webm', realtime: true, width: 640, idleAfterMs: 1000 });
  stream.pipe(res);
  stream.on('end', () => avatar.destroy());
  req.on('close', () => { stream.destroy(); avatar.destroy(); });
}).listen(8091, () => console.log('http://localhost:8091/'));
```

- `format` は `'webm'`・`'fmp4'`・`'jpeg'`。

## 文章を話させる（ZEN-TTS）

ブラウザ：

```html
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>ZEN-LipSync + ZEN-TTS</title>
  <script type="importmap">
    { "imports": {
        "@zenryokukikai/zen-lipsync": "/node_modules/@zenryokukikai/zen-lipsync/dist/index.js",
        "@zenryokukikai/zen-lipsync/tts": "/node_modules/@zenryokukikai/zen-lipsync/dist/tts/index.js",
        "zen-tts": "/node_modules/zen-tts/dist/index.js"
    } }
  </script>
</head>
<body>
  <div id="avatar" style="max-width: 640px"></div>
  <button id="talk">話させる</button>
  <p>音声合成：ZEN-TTS / リップシンク：ZEN-LipSync（<a href="https://zenryokukikai.com">https://zenryokukikai.com</a>）</p>

  <script type="module">
    import { create } from '@zenryokukikai/zen-lipsync';
    import { speakText } from '@zenryokukikai/zen-lipsync/tts';
    import { ZenTts, ztmUrl } from 'zen-tts';

    document.getElementById('talk').onclick = async () => {
      const [avatar, tts] = await Promise.all([
        create({
          container: document.getElementById('avatar'),
          assetsBaseUrl: '/node_modules/zen-lipsync-avatars/',
          avatar: 'avatars/marine/avatar.json',
        }),
        ZenTts.load({
          model: ztmUrl('marine'),                      // 声：全力機械の CDN の marine.ztm
          modelsUrl: '/node_modules/zen-tts/models',    // 読みのモデル（g2p/）
          cpuWasmUrl: '/node_modules/zen-tts/cpu/tts-cpu.wasm',
        }),
      ]);
      await speakText(avatar, tts, 'こんにちは。全力マリンです。今日はいい天気ですね。', { emotion: { happy: 60, neutral: 40 } });
    };
  </script>
</body>
</html>
```

Node.js（文章 → MP4）：

```js
import { create } from '@zenryokukikai/zen-lipsync/node';
import { renderText } from '@zenryokukikai/zen-lipsync/tts/node';
import { ZenTtsNode, ztmUrl } from 'zen-tts/node';

const [avatar, tts] = await Promise.all([
  create({ avatar: 'marine', assetsPath: 'node_modules/zen-lipsync-avatars' }),
  ZenTtsNode.load({ model: ztmUrl('marine') }),   // 声：全力機械の CDN の marine.ztm（読みのモデルはパッケージに入っています）
]);
const r = await renderText(avatar, tts, 'こんにちは。全力マリンです。今日はいい天気ですね。', {
  out: 'marine-tts.mp4',
  tts: { emotion: { happy: 60, neutral: 40 } },
});
console.log(`${r.file}：${(r.durationMs / 1000).toFixed(2)} 秒`);
avatar.destroy();
```

- 声は `marine`・`hashiro`・`kai`・`kikka`（`ztmUrl('kikka')` など）。オプションは ZEN-TTS の `speak()` と同じです（`emotion`・`speed`・`pitch` など）。

## 会話

文字で話しかけると、くり返して答えるいちばん小さい例です。

```html
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>ZEN-LipSync：会話</title>
  <script type="importmap">
    { "imports": {
        "@zenryokukikai/zen-lipsync": "/node_modules/@zenryokukikai/zen-lipsync/dist/index.js",
        "@zenryokukikai/zen-lipsync/conversation": "/node_modules/@zenryokukikai/zen-lipsync/dist/conversation/index.js",
        "zen-tts": "/node_modules/zen-tts/dist/index.js"
    } }
  </script>
</head>
<body>
  <div id="avatar" style="max-width: 640px"></div>
  <form id="form"><input id="text" value="こんにちは" size="40"> <button>話しかける</button></form>
  <p>音声合成：ZEN-TTS / リップシンク：ZEN-LipSync（<a href="https://zenryokukikai.com">https://zenryokukikai.com</a>）</p>

  <script type="module">
    import { create } from '@zenryokukikai/zen-lipsync';
    import { Conversation } from '@zenryokukikai/zen-lipsync/conversation';
    import { ZenTts, ztmUrl } from 'zen-tts';

    let conversation = null;
    document.getElementById('form').onsubmit = async (e) => {
      e.preventDefault();
      if (!conversation) {
        const [avatar, tts] = await Promise.all([
          create({ container: document.getElementById('avatar'), assetsBaseUrl: '/node_modules/zen-lipsync-avatars/', avatar: 'avatars/kikka/avatar.json' }),
          ZenTts.load({ model: ztmUrl('kikka'), modelsUrl: '/node_modules/zen-tts/models', cpuWasmUrl: '/node_modules/zen-tts/cpu/tts-cpu.wasm' }),
        ]);
        conversation = new Conversation({
          avatar,
          // 返事：ここでは、言われたことをくり返すだけ（LLM の返事を少しずつ yield すれば、そのまま会話になります）
          async *reply({ messages }) {
            yield `「${messages.at(-1).content}」ですね。`;
          },
          // 声：返事の文を ZEN-TTS で音にする
          async *synthesize({ piece }) {
            for await (const c of tts.speakStream(piece)) yield { pcm: c.pcm, sampleRate: c.sampleRate };
          },
        });
        conversation.start();
      }
      conversation.sendText(document.getElementById('text').value);   // マイクを使うときは openMic({ conversation })
    };
  </script>
</body>
</html>
```

- LLM につなぐときは、`reply()` で返事を少しずつ `yield` します。マイクは `openMic({ conversation })`。

## もっと

- [API.md](API.md)：関数・オプション・イベントの一覧
- [README](../README.md)：速さ・メモリ・Node.js のネイティブ版など
- zen-lipsync-avatars の `examples/`：音声認識つきの会話など（`pnpm fetch-asr` と `pnpm fetch-tts-models` でモデルを取ってから）
