// docs/API.md from the TypeScript declarations that `pnpm build` writes to dist/ (no extra dependency).
//   pnpm build && node scripts/gen-api-md.mjs
// For each entry point (package.json "exports"), the public names listed in ENTRIES are shown with their declaration
// (as in the .d.ts, private members left out) and a one-line Japanese description (DESC_JA). Every other name the entry
// exports is listed under 「低レベル」. A name exported by an entry but missing from its .d.ts files fails the script,
// so the page cannot drift from the code.
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const dist = path.join(root, 'dist');
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));

/** The public surface per entry: [entry, import path, title, [names]]. Order is the order on the page. */
const ENTRIES = [
  ['.', '@zenryokukikai/zen-lipsync', 'ブラウザ（WebGPU）', ['create', 'BrowserCreateOptions', 'CreateOptions', 'AvatarFit', 'AvatarHandle', 'SpeechAudio', 'AvatarEvents', 'LoadProgress', 'FrameTiming', 'LipWeights', 'WebGPUUnavailableError', 'OptionsError', 'AvatarPackError', 'VERSION']],
  ['./react', '@zenryokukikai/zen-lipsync/react', 'React', ['ZenLipSync', 'ZenLipSyncProps']],
  ['./node', '@zenryokukikai/zen-lipsync/node', 'Node.js（CPU）', ['create', 'NodeCreateOptions', 'NodeAvatar', 'FramesOptions', 'RenderOptions', 'StreamOptions', 'NodeFrame', 'RenderProgress', 'renderAudioFile', 'RenderAudioFileOptions', 'RenderAudioResult', 'renderAudio', 'openWav', 'WavReader', 'FfmpegMissingError', 'WavFormatError']],
  ['./tts', '@zenryokukikai/zen-lipsync/tts', 'ZEN-TTS で話させる（ブラウザ）', ['speakText', 'SpeakTextOptions', 'TtsEngine', 'splitSentences', 'TTS_SAMPLE_RATE']],
  ['./tts/node', '@zenryokukikai/zen-lipsync/tts/node', 'ZEN-TTS で話させる（Node.js）', ['renderText', 'streamText', 'speakTextFrames', 'TextOutputOptions']],
  ['./conversation', '@zenryokukikai/zen-lipsync/conversation', '会話（マイク → 返事 → 声 → アバター）', ['Conversation', 'ConversationOptions', 'ReplyContext', 'SynthesisContext', 'ConversationMessage', 'ConversationEvents', 'ConversationStats', 'TurnLatency', 'EmotionReading', 'openMic', 'MicOptions', 'Mic']],
];

/** One-line Japanese descriptions of the public names (the .d.ts comments are in English and kept below them). */
const DESC_JA = {
  create: 'アバターを読み込み、待機映像を流し始めます。',
  'node:create': 'アバターを CPU に読み込みます（ffmpeg が必要）。',
  BrowserCreateOptions: 'ブラウザの create() のオプション。',
  CreateOptions: 'create() のオプション。',
  AvatarFit: '枠への収め方（既定 contain）。',
  AvatarHandle: '読み込んだアバター。',
  SpeechAudio: '音声：モノラルの Float32Array とサンプリングレート。',
  AvatarEvents: 'on() のイベント。',
  LoadProgress: '読み込みの進み具合。',
  FrameTiming: '1 フレームの推論時間。',
  LipWeights: 'モデルの重みの形式（既定 i8）。',
  WebGPUUnavailableError: 'WebGPU が使えないとき。',
  OptionsError: 'オプションが正しくないとき。',
  AvatarPackError: 'アバターのファイルを読めないとき。',
  VERSION: 'SDK のバージョン。',
  ZenLipSync: 'アバターを置くコンポーネント。',
  ZenLipSyncProps: 'props。',
  NodeCreateOptions: 'create() のオプション。',
  NodeAvatar: '音声から frames()・render()・stream() を作ります。',
  FramesOptions: 'frames() などの共通オプション。',
  RenderOptions: 'render() のオプション。',
  StreamOptions: 'stream() のオプション。',
  NodeFrame: 'frames() の 1 フレーム。',
  RenderProgress: 'render() の進み具合。',
  renderAudioFile: 'WAV ファイル → 動画ファイル。',
  RenderAudioFileOptions: 'renderAudioFile() のオプション。',
  RenderAudioResult: 'renderAudioFile() の結果。',
  renderAudio: '読み込み済みのアバターで音声 → 動画。',
  openWav: 'WAV を少しずつ読みます。',
  WavReader: 'openWav() の戻り値。',
  FfmpegMissingError: 'ffmpeg がないとき。',
  WavFormatError: '読めない WAV のとき。',
  speakText: '文章を ZEN-TTS で話させます。',
  SpeakTextOptions: 'speakText() のオプション。',
  TtsEngine: 'ZenTts／ZenTtsNode の形。',
  splitSentences: '文章を文に分けます。',
  TTS_SAMPLE_RATE: '44100。',
  renderText: '文章 → 動画ファイル。',
  streamText: '文章 → 配信。',
  speakTextFrames: '文章 → フレーム。',
  TextOutputOptions: 'tts オプション。',
  Conversation: '会話の進行役。',
  ConversationOptions: 'Conversation のオプション。',
  ReplyContext: 'reply() の引数。',
  SynthesisContext: 'synthesize() の引数。',
  ConversationMessage: '1 発言。',
  ConversationEvents: 'on() のイベント。',
  ConversationStats: 'stats() の戻り値。',
  TurnLatency: '1 ターンごとの遅れ。',
  EmotionReading: 'emotion() の戻り値。',
  openMic: 'マイクの話し始め・話し終わりを伝えます。',
  MicOptions: 'openMic() のオプション。',
  Mic: 'openMic() の戻り値。',
};

// ---- .d.ts reading -------------------------------------------------------------------------------------------------

/** Exports of a .d.ts file, followed through `export * from` and `export { a as b } from`: name -> { file, local }. */
function exportsOf(file, seen = new Set()) {
  const out = new Map();
  if (seen.has(file)) return out;
  seen.add(file);
  const src = fs.readFileSync(file, 'utf8');
  const rel = (p) => path.resolve(path.dirname(file), p.replace(/\.js$/, '.d.ts'));
  for (const m of src.matchAll(/^export \* from '([^']+)';/gm)) for (const [k, v] of exportsOf(rel(m[1]), seen)) if (!out.has(k)) out.set(k, v);
  for (const m of src.matchAll(/^export (?:type )?\{([^}]*)\}(?: from '([^']+)')?;/gm)) {
    for (const raw of m[1].split(',')) {
      const part = raw.trim().replace(/^type\s+/, '');
      if (!part) continue;
      const [local, as] = part.split(/\s+as\s+/);
      const name = (as ?? local).trim();
      if (m[2]) {
        const target = rel(m[2]);
        const sub = exportsOf(target, new Set());
        out.set(name, sub.get(local.trim()) ?? { file: target, local: local.trim() });
      } else out.set(name, { file, local: local.trim() });
    }
  }
  for (const m of src.matchAll(/^export (?:declare )?(?:abstract )?(?:class|function|interface|type|const|enum) ([A-Za-z0-9_$]+)/gm)) out.set(m[1], { file, local: m[1] });
  return out;
}

/** The declaration of `name` in a .d.ts: its JSDoc and its text, private members and blank lines removed. */
function declarationOf(file, name) {
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  const re = new RegExp(`^export (?:declare )?(?:abstract )?(class|function|interface|type|const) ${name.replace('$', '\\$')}\\b`);
  const starts = [];
  lines.forEach((l, i) => { if (re.test(l)) starts.push(i); });
  if (!starts.length) return null;
  const parts = [];
  let doc = '';
  for (const s of starts) {
    // JSDoc right above
    let d = s - 1;
    if (d >= 0 && lines[d].trim().endsWith('*/')) {
      while (d > 0 && !lines[d].trim().startsWith('/**')) d--;
      if (!doc) doc = lines.slice(d, s).join('\n');
    }
    // the declaration's own lines: until braces balance and the statement ends
    let depth = 0, e = s;
    for (; e < lines.length; e++) {
      for (const ch of lines[e]) { if (ch === '{' || ch === '(') depth++; else if (ch === '}' || ch === ')') depth--; }
      if (depth <= 0 && /[;}]\s*$/.test(lines[e])) break;
    }
    parts.push(lines.slice(s, e + 1));
  }
  const text = parts.map((ls) => dropPrivate(ls).map((l) => l.replace(/^export (declare )?/, '')).join('\n')).join('\n');
  return { doc: cleanDoc(doc), text };
}

/** The declaration lines without private members and the constructor, and without the JSDoc block right above them. */
function dropPrivate(ls) {
  const hidden = (l) => /^\s+private\b/.test(l) || /^\s+constructor\(/.test(l);
  const keep = ls.map(() => true);
  for (let i = 0; i < ls.length; i++) {
    if (ls[i].trim() === '') { keep[i] = false; continue; }
    if (!hidden(ls[i])) continue;
    keep[i] = false;
    let j = i - 1;
    if (j >= 0 && ls[j].trim().endsWith('*/')) { while (j >= 0) { keep[j] = false; if (ls[j].trim().startsWith('/**')) break; j--; } }
  }
  return ls.filter((_, i) => keep[i]);
}

/** One line from a JSDoc block. */
function cleanDoc(doc) {
  const t = doc.replace(/^\s*\/\*\*|\*\/\s*$/g, '').split('\n').map((l) => l.replace(/^\s*\* ?/, '')).join(' ').replace(/\s+/g, ' ').trim();
  return t;
}

// ---- the page -----------------------------------------------------------------------------------------------------

/** Members not meant for apps (internals the classes expose for the SDK's own use). */
const HIDE_MEMBERS = new Set(['lips', 'set', 'memoryBytes', 'now']);

/** A one-line signature: functions as `name(params): ret`, interfaces / classes as their member names, types as `= …`. */
function signature(name, text) {
  const flat = text.replace(/\/\*\*[\s\S]*?\*\//g, '').replace(/\s+/g, ' ').trim();
  let m = flat.match(/^function \w+(\(.*\)): (.+?);?$/);
  if (m) {
    const sig = `${name}${m[1]}: ${m[2]}`.replace(/;$/, '');
    // long return types (inline object types) shortened to keep one line per row
    return sig.length <= 110 ? sig : `${name}${m[1]}`.length <= 110 ? `${name}${m[1]}` : `${name}(${m[1].slice(1, -1).split(/,\s*(?![^<{(]*[>})])/).map((x) => x.split(':')[0].trim()).join(', ')})`;
  }
  m = flat.match(/^const \w+(:.*?)?(=.*)?;$/);
  if (m) return `${name}${m[1] ?? ''}`;
  m = flat.match(/^type \w+(<[^=]*>)? = (.*);$/);
  if (m) return `${name} = ${m[2]}`;
  if (/^(class|interface) /.test(flat)) {
    // member names at depth 1 of the original text (one member starts per line; arrow types do not count as brackets)
    const names = [];
    let depth = 0;
    for (const line of text.replace(/\/\*\*[\s\S]*?\*\//g, '').split('\n')) {
      if (depth === 1) {
        const t = line.trim();
        const getter = /^(readonly )?get /.test(t);
        const mm = t.replace(/^(readonly |get |static |async )+/, '').match(/^([A-Za-z0-9_$]+)(\??)(\(|<|:)/);
        if (mm && !HIDE_MEMBERS.has(mm[1])) names.push(mm[1] + mm[2] + (!getter && (mm[3] === '(' || mm[3] === '<') ? '()' : ''));
      }
      for (const ch of line) { if ('({['.includes(ch)) depth++; else if (')}]'.includes(ch)) depth--; }
    }
    const ext = flat.match(/^(?:class|interface) \w+[^{]*?\b(extends|implements) ([^{]+?)\s*\{/);
    const list = [...new Set(names)].join(', ');
    if (ext && ext[1] === 'extends') return list ? `${ext[2]} + { ${list} }` : `extends ${ext[2]}`;
    return `{ ${list} }`;
  }
  return flat.length > 120 ? flat.slice(0, 117) + '…' : flat;
}

const esc = (t) => t.replace(/\|/g, '\\|').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const out = [];
out.push(`# ZEN-LipSync SDK API（${pkg.version}）`, '');
out.push('組み込み方は [GUIDE.md](GUIDE.md)。この一覧は `node scripts/gen-api-md.mjs` で、型定義から作っています。', '');
for (const [entry, imp, title, names] of ENTRIES) {
  const exp = pkg.exports[entry];
  const dts = path.join(root, (typeof exp === 'string' ? exp : exp.types).replace(/^\.\//, ''));
  const all = exportsOf(dts);
  out.push(`## ${title}`, '', `\`${imp}\``, '', '| 名前 | 形 | 説明 |', '|---|---|---|');
  for (const name of names) {
    const where = all.get(name);
    if (!where) throw new Error(`${imp}: "${name}" is not exported (update ENTRIES)`);
    const d = declarationOf(where.file, where.local);
    if (!d) throw new Error(`${imp}: no declaration of "${where.local}" in ${path.relative(root, where.file)}`);
    const ja = DESC_JA[`${entry.replace('./', '')}:${name}`] ?? DESC_JA[name];
    if (!ja) throw new Error(`${imp}: no Japanese description for "${name}" (add it to DESC_JA)`);
    out.push(`| \`${name}\` | <code>${esc(signature(name, d.text))}</code> | ${ja} |`);
  }
  out.push('');
  const rest = [...all.keys()].filter((n) => !names.includes(n)).sort();
  if (rest.length) out.push('<details><summary>低レベル</summary>', '', rest.map((n) => `\`${n}\``).join('、'), '', '</details>', '');
}

fs.writeFileSync(path.join(root, 'docs', 'API.md'), out.join('\n'));
console.log('docs/API.md written');
