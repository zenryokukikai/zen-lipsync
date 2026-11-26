# アバターパックの形式 / Avatar pack format (version 3 and 2; version 1 is still read)

ZEN-LipSync は、アバター 1 人ぶんのファイルをまとめた「アバターパック」を読み込んで動きます。
ZEN-LipSync renders an avatar from an *avatar pack*: one directory with the files below.

> **重要：** リップシンクのモデルは、アバターの待機映像ごとに用意したものです。パックは全力機械が作成します。
> 新しいアバター（ご自身の顔・映像）のモデル作成は、全力機械にご依頼ください（https://zenryokukikai.com ）。
> **Important:** the lip-sync models are made for each avatar's idle video, and packs are made by Zenryoku Kikai.
> To have a model made for a new avatar (your own face or video), please contact Zenryoku Kikai (https://zenryokukikai.com).

## アバター（avatar.json）/ Avatar (avatar.json)

1 人のアバターは、待機映像ごとのパックと、共通のモデルからなります（例：[zen-lipsync-avatars](https://github.com/zenryokukikai/zen-lipsync-avatars) の `avatars/marine/`）。
An avatar is one pack per idle video plus the models they share (example: `avatars/marine/` in [zen-lipsync-avatars](https://github.com/zenryokukikai/zen-lipsync-avatars)).

```json
{
  "format": "zen-lipsync-avatar-set", "version": 1, "id": "marine",
  "name": { "ja": "全力 マリン", "en": "Zenryoku Marine" }, "model": "marine-1.0",
  "sources": { "normal": "normal", "angry": "angry" }, "default": "normal",
  "idleSwitch": "idle-switch.json"
}
```

`model`（省略可）：アバターのモデル一式のバージョンです（例：`marine-1.0`）。読み込みには使いません。
`model` (optional): the version of the avatar's model set (e.g. `marine-1.0`); not used for loading.

`idle-switch.json`：`{ "version": 1, "sources": [...], "pairs": { "normal->angry": { "to": [...], "dist": [...] }, ... } }`。
`to[u]` は、元の待機映像の元フレーム u にいちばん似た、切り替え先の元フレーム。`dist[u]` はその差（平均 |RGB|）。
`to[u]` is the target loop's source frame most like source frame u; `dist[u]` their difference (mean |RGB|).

## パック / Pack

## ファイル / Files

| ファイル / File | 内容 / Contents |
|---|---|
| `manifest.json` | 下の形式。パックの説明と、全ファイルの大きさと SHA-256 / The description below, with every file's size and SHA-256 |
| `idle.webm`（`idle.mp4`） | 待機映像（25 fps、`walk` の順にフレームが並ぶ）/ The idle video (25 fps; its frames follow `walk`) |
| `frame.i8.wgw` | フレームモデルの重み（int8、WebGPU 用）/ Frame model weights (int8, WebGPU) |
| `frame.wgw` | フレームモデルの重み（float32、任意）/ Frame model weights (float32, optional) |
| `frame.lip8` | フレームモデルの重み（int8、CPU 用）/ Frame model weights (int8, CPU paths) |
| `sentence.wgw` | 文モデルの重み（float32）/ Sentence model weights (float32) |

モデルの 4 ファイルは、`manifest.json` の `models`（パックからの相対パス）の場所にも置けます。マリンは `../models` を共有しています。
The model files may live in the directory given by `models` (relative to the pack); Marine's packs share `../models`.
| `features.f16` | 元フレームごとの特徴（fp16、`source_frames × 32 × 20 × 44`）/ Features per source frame |
| `ref_latent.f32` | 元フレームごとの参照（float32、`source_frames × 128`）/ Reference latent per source frame |
| `mouth_open_dir.f32` | 任意。口が開く向き（float32 128 個、長さ 1）。口の開き具合 `mouthOpen` に要る（無いパックでは使えない）/ Optional: the direction the mouth-motion latent opens the mouth along (128 float32, unit length); `mouthOpen` needs it |
| `mouth_alpha.u8` | バージョン 3：元フレームごとの口のマスク（`uint8`、`source_frames × (y1 − y0) × (x1 − x0)`、`box` の範囲）。描いた口をどれだけ重ねるか / Version 3: the mouth mask per source frame (`uint8`, `source_frames × (y1 − y0) × (x1 − x0)`, the `box` area): how much of the painted mouth is drawn |
| `tone.json` | 任意。色の対応表 `{ "from_video": [3][256], "to_video": [3][256] }`。無ければ、モデルの変化をそのまま足す（新しいパックは入れない）/ Optional colour tables; without it the model change is added as it is (new packs leave it out) |
| `frames.pack` | バージョン 2：全元フレームの口元の画像とマスクを 1 ファイルにまとめたもの（下の「frames.pack」）/ Version 2: every source frame's mouth crop and mask in one file (see "frames.pack") |
| `frames/crop_NNN.png` | バージョン 1：元フレームごとの口元の画像（512 の顔切り出しのうち `crop_box` の範囲）/ Version 1: mouth crop per source frame |
| `frames/alpha_NNN.png` | バージョン 1：重ねるときのマスク（グレー、`rect` の大きさ）/ Version 1: patch mask per source frame |
| `matte.webm` | 任意。背景を抜いた表示（`create({ matte: 'matte.webm' })`・`createRenderer({ matte })`）に使う縦 2 段の映像。上半分は人物の色（髪や輪郭の半透明の所は背景の色を除いた色）、下半分は人物の形（グレー、白＝人物）。待機映像と同じ幅、高さは 2 倍、同じフレーム数・25 fps、VP9 / Optional: the stacked video for the cut-out (`create({ matte: 'matte.webm' })`, `createRenderer({ matte })`): top half the person's colour (with the background's colour taken out where the person is see-through), bottom half the person's matte (grey, white = person); the idle video's width, twice its height, the same frames at 25 fps, VP9 |
| `sample.wav` | 試し用の音声（任意）/ A sample speech clip (optional) |

`.wgw`：`uint32` のヘッダー長（リトルエンディアン）、JSON `{"tensors": {name: {offset, shape, dtype?}}}`、データ。
`.wgw`: a little-endian `uint32` header length, a JSON header `{"tensors": {name: {offset, shape, dtype?}}}`, then the data.

## manifest.json

```json
{
  "format": "zen-lipsync-avatar",
  "version": 2,
  "name": "marine-normal",
  "frame_size": [1264, 720],
  "fps": 25,
  "source_frames": 251,
  "walk": [0, 1, 2, "…", 250, 249, "…", 1],
  "geo": { "0": { "rect": [x, y, w, h], "matrix": [[a, b, c], [d, e, f]], "abs_x": 542, "abs_y": 181, "crop_box": [x0, y0, x1, y1] } },
  "box": { "y0": 312, "y1": 472, "x0": 72, "x1": 424 },
  "feat_dims": [32, 20, 44],
  "latent_window": 5,
  "video": [{ "src": "idle.webm", "type": "video/webm; codecs=\"vp9\"" }],
  "files": { "../models/frame.i8.wgw": { "bytes": 2686544, "sha256": "…" } },
  "models": "../models",
  "frames": { "pack": "frames.pack", "type": "image/webp" },
  "license": "…",
  "credit": "…"
}
```

- `walk`：動画のフレーム k が表示する元フレームの番号（往復ループ）。/ Source frame shown at video frame k (a ping-pong loop).
- `geo`：元フレームごとに、口元の画像を動画のどこへ重ねるか。/ Where each source frame's mouth patch goes on the video.
- `box`：フレームモデルが出力する口元の範囲（512 の顔切り出しの座標）。/ The frame model's mouth box in the 512 face crop.
- `frames`（バージョン 2・3）：口元の画像とマスクをまとめたファイルと、中の画像の形式。ファイルは `files` にも載せる。/ Versions 2 and 3: the file with the mouth crops and masks, and the image type inside; the file is also listed in `files`.
- バージョン 1 のパック（`frames` が無く、`frames/` の PNG を使う）も、そのまま読み込めます。/ Version 1 packs (no `frames`; the PNG files in `frames/`) still load unchanged.

## バージョン 3（口を描くモデル）/ Version 3 (a model that paints the mouth)

バージョン 2 の形に、次の 3 つを足したものです。ZEN-LipSync 1.5.0 以降で読めます（1.4.0 以前は「version 1 or 2」のエラーで読み込みを止めます）。
Version 2 plus the three items below. Read by ZEN-LipSync 1.5.0 and later (1.4.0 and earlier stop with a "version 1 or 2" error).

- `"frame_output": "absolute"`：フレームモデルが口元そのものを描く。描いた口は、待機映像のフレームにそのまま重ねる：`out = 映像 × (1 − A) + 描いた口 × A`（`A` = 重ねるマスク × `mouth_alpha.u8` × 四角の端の窓 × 話し終わりの薄め）。口元の画像（crop）は使わない。/ The frame model paints the mouth itself; it is drawn over the idle video frame as it is (`A` = patch mask × `mouth_alpha.u8` × the box's edge window × the fade after speech). The crops take no part.
- `"model_colour"`：`[3][256]`、映像の値 → モデルの色（チャンネルごとに単調）。描いた色はこの逆で映像の色に戻す。同梱のアバターは恒等（0, 1, …, 255）。/ `[3][256]`, video value → the model's colour (monotone per channel); the painted colour goes back through its inverse. The bundled avatars use the identity.
- `mouth_alpha.u8`（`files` に載せる）。`tone.json` は持たない。/ `mouth_alpha.u8` (listed in `files`); no `tone.json`.
- 描いた口は、モデルの四角から映像へ bicubic（OpenCV の INTER_CUBIC と同じ、a = −0.75）で戻す。/ The painted mouth is taken from the model's box into the frame bicubic (OpenCV INTER_CUBIC's kernel, a = −0.75).
- `frame_output` が `"absolute"` なのはバージョン 3 だけで、バージョン 3 は必ず `"absolute"`。/ `"absolute"` goes with version 3 only, and version 3 is always `"absolute"`.

## frames.pack（バージョン 2・3 / versions 2 and 3）

1 本の待機映像の、全元フレームの口元の画像（RGB）とマスク（グレー）を、1 ファイルにまとめたものです。1 回の取得で全部そろい、ブラウザは届いた分から使います。
All source frames' mouth crops (RGB) and masks (grey) of one idle loop in one file: one download, and the browser uses frames as their bytes arrive.

| 位置 / Offset | 内容 / Contents |
|---|---|
| 0–7 | `ZLFPACK1`（ASCII） |
| 8–11 | 目次の長さ L（`uint32`、リトルエンディアン）/ Index length L (little-endian `uint32`) |
| 12–(12+L) | 目次（UTF-8 の JSON）/ Index (UTF-8 JSON): `{ "type": "image/webp", "frames": [[cropOffset, cropLength, alphaOffset, alphaLength], …] }` |
| 以降 / then | 画像。元フレーム 0, 1, 2, … の順に、口元の画像、マスクの順 / The images: source frame 0, 1, 2, …, each crop then mask |

- `frames[u]` は元フレーム u の、口元の画像とマスクの位置（ファイルの先頭からのバイト数）と長さ。/ `frames[u]`: byte offset (from the start of the file) and length of source frame u's crop and mask.
- 画像は WebP（品質 90）。口元の画像は `crop_box` の大きさの RGB、マスクは `rect` の大きさのグレー（バージョン 1 の PNG と同じ絵）。/ The images are WebP (quality 90): the crop is RGB of the `crop_box` size, the mask grey of the `rect` size (the same pictures as the version 1 PNG files).
