# アバターパックの形式 / Avatar pack format (version 2; version 1 is still read)

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

`model`（省略可）：アバターのモデル一式の版です（例：`marine-1.0`）。読み込みには使いません。
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
| `tone.json` | 色の対応表 `{ "from_video": [3][256], "to_video": [3][256] }` / Colour tables |
| `frames.pack` | 版 2：全元フレームの口元の画像とマスクを 1 ファイルにまとめたもの（下の「frames.pack」）/ Version 2: every source frame's mouth crop and mask in one file (see "frames.pack") |
| `frames/crop_NNN.png` | 版 1：元フレームごとの口元の画像（512 の顔切り出しのうち `crop_box` の範囲）/ Version 1: mouth crop per source frame |
| `frames/alpha_NNN.png` | 版 1：重ねるときのマスク（グレー、`rect` の大きさ）/ Version 1: patch mask per source frame |
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
- `frames`（版 2）：口元の画像とマスクをまとめたファイルと、中の画像の形式。ファイルは `files` にも載せる。/ Version 2: the file with the mouth crops and masks, and the image type inside; the file is also listed in `files`.
- 版 1 のパック（`frames` が無く、`frames/` の PNG を使う）も、そのまま読み込めます。/ Version 1 packs (no `frames`; the PNG files in `frames/`) still load unchanged.

## frames.pack（版 2 / version 2）

1 本の待機映像の、全元フレームの口元の画像（RGB）とマスク（グレー）を、1 ファイルにまとめたものです。1 回の取得で全部そろい、ブラウザは届いた分から使います。
All source frames' mouth crops (RGB) and masks (grey) of one idle loop in one file: one download, and the browser uses frames as their bytes arrive.

| 位置 / Offset | 内容 / Contents |
|---|---|
| 0–7 | `ZLFPACK1`（ASCII） |
| 8–11 | 目次の長さ L（`uint32`、リトルエンディアン）/ Index length L (little-endian `uint32`) |
| 12–(12+L) | 目次（UTF-8 の JSON）/ Index (UTF-8 JSON): `{ "type": "image/webp", "frames": [[cropOffset, cropLength, alphaOffset, alphaLength], …] }` |
| 以降 / then | 画像。元フレーム 0, 1, 2, … の順に、口元の画像、マスクの順 / The images: source frame 0, 1, 2, …, each crop then mask |

- `frames[u]` は元フレーム u の、口元の画像とマスクの位置（ファイルの先頭からのバイト数）と長さ。/ `frames[u]`: byte offset (from the start of the file) and length of source frame u's crop and mask.
- 画像は WebP（品質 90）。口元の画像は `crop_box` の大きさの RGB、マスクは `rect` の大きさのグレー（版 1 の PNG と同じ絵）。/ The images are WebP (quality 90): the crop is RGB of the `crop_box` size, the mask grey of the `rect` size (the same pictures as the version 1 PNG files).
