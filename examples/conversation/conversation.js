// Real-time conversation sample for @zenryokukikai/zen-lipsync/conversation.
// mic -> VAD/EndPointer turn end -> ASR (Web Speech API) -> scripted reply (stand-in for an LLM) -> ZEN-TTS
// (streaming) -> avatar lip-sync. Everything runs in the browser; no audio or video leaves the device.
//
// Layout it looks for (relative to this page; each can be overridden with a query parameter):
//   SDK      ./sdk/dist/            else ../../dist/ (this repository after `pnpm build`)      ?sdk=<dir>
//   avatars  ./avatars/<id>/avatar.json, list from ./avatars/index.json                        ?assets=<dir>  ?avatars=a,b,c
//            (in this repository: ?assets=<your zen-lipsync-avatars checkout>/ ; without index.json the list is
//            marine, hashiro, kikka)
//   ZEN-TTS  ./tts/dist/index.js with its models in ./models/ (models/manifest.json)           ?tts=<module url>  ?models=<dir>
// Other queries: ?avatar=<id>  ?voice=<ZEN-TTS voice id>  ?ttsbackend=auto|webgpu|cpu

const V = '1';   // cache-buster for this page's own modules (not for packs or models)
const q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const dir = (p) => (p.endsWith('/') ? p : p + '/');

// --- log / status ---------------------------------------------------------------------------------------------
const logEl = $('log');
const log = (kind, detail) => {
  const line = `${new Date().toLocaleTimeString('ja-JP')} ${kind}${detail === undefined ? '' : ' ' + JSON.stringify(detail)}`;
  logEl.textContent += line + '\n';
  logEl.scrollTop = logEl.scrollHeight;
};
const say = (text) => { $('state').textContent = text; };
const fail = (text) => { $('err').textContent = text; };

// --- SDK ------------------------------------------------------------------------------------------------------
async function importFirst(candidates, file) {
  let last;
  for (const base of candidates) {
    try { return { mod: await import(`${dir(base)}${file}?v=${V}`), base: dir(base) }; } catch (e) { last = e; }
  }
  throw last;
}
const sdkDirs = q.get('sdk') ? [q.get('sdk')] : ['./sdk/dist/', '../../dist/'];
const { mod: sdk, base: SDK } = await importFirst(sdkDirs, 'index.js');
const { Conversation, openMic } = await import(`${SDK}conversation/index.js?v=${V}`);
const { create, WebGPUUnavailableError } = sdk;

// --- avatar and voice lists (from the files, not hard-coded) --------------------------------------------------
const ASSETS = dir(q.get('assets') ?? './');
const TTS_URL = q.get('tts') ?? './tts/dist/index.js';
const MODELS = dir(q.get('models') ?? './models/');
const BACKEND = q.get('ttsbackend') ?? 'auto';
async function fetchJson(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
const avatarIds = q.get('avatars') ? q.get('avatars').split(',').filter(Boolean)
  : await fetchJson(`${ASSETS}avatars/index.json`).catch(() => ['marine', 'hashiro', 'kikka']);
const AVATAR_LIST = (await Promise.all(avatarIds.map((id) => fetchJson(`${ASSETS}avatars/${id}/avatar.json`)
  .then((a) => ({ id, name: a.name?.ja ?? id }), () => null)))).filter(Boolean);
const VOICE_LIST = await fetchJson(`${MODELS}manifest.json`)
  .then((m) => Object.entries(m.voices ?? {}).map(([id, v]) => ({ id, name: v.name?.ja ?? id })), () => []);
const AVATARS = AVATAR_LIST.map((a) => a.id);
const VOICES = VOICE_LIST.map((v) => v.id);
const AVATAR = AVATARS.includes(q.get('avatar')) ? q.get('avatar') : (AVATARS.includes('marine') ? 'marine' : AVATARS[0]);
const VOICE = VOICES.includes(q.get('voice')) ? q.get('voice') : (VOICES.includes(AVATAR) ? AVATAR : VOICES[0]);

// --- reply: a script streamed sentence by sentence (replace with your LLM) ------------------------------------
const SCRIPT = [
  'こんにちは。全力機械のアバターです。今日は何について話しましょうか。',
  'なるほど。それは面白いですね。もう少し詳しく教えてもらえますか。',
  'ありがとうございます。整理すると、三つの点がありそうです。',
  'その点は私も気になっていました。少し違う見方もできます。',
  '確認できました。次はどうしましょうか。',
];
let turn = 0;
async function* reply({ signal }) {
  const text = SCRIPT[turn++ % SCRIPT.length];
  for (const piece of text.split(/(?<=。)/)) {
    if (signal.aborted) return;
    await new Promise((r) => setTimeout(r, 60));   // token streaming; the first piece sets speechEnd -> first token
    yield piece;
  }
}

// --- voice: ZEN-TTS streaming -----------------------------------------------------------------------------
let tts = null;
async function loadTts() {
  say(`ZEN-TTS（${VOICE}）を読み込んでいます…`);
  try {
    const mod = await import(`${TTS_URL}?v=${V}`);
    const ttsDir = new URL('.', new URL(TTS_URL, location.href)).href;
    tts = await mod.ZenTts.load({ modelsUrl: MODELS, voice: VOICE, backend: BACKEND, cpuWasmUrl: new URL('../cpu/tts-cpu.wasm', ttsDir).href });
    log('TTS ready', { voice: tts.voice, backend: tts.backend });
  } catch (e) {
    tts = null;
    fail(`ZEN-TTS を読み込めませんでした: ${String(e && e.message || e)}`);
    log('TTS failed', { error: String(e) });
    return false;
  }
  return true;
}
async function* synthesize({ piece, signal }) {
  for await (const c of tts.speakStream(piece)) {
    if (signal.aborted) return;
    yield { pcm: c.pcm, sampleRate: c.sampleRate ?? 44100 };
  }
}

// --- avatar and conversation ----------------------------------------------------------------------------------
let avatar = null;
let conversation = null;
function placeholder() {
  const p = document.createElement('div');
  p.className = 'placeholder';
  p.textContent = '「開始」でアバターと ZEN-TTS を読み込みます';
  $('stage').replaceChildren(p);
}
async function boot() {
  $('start').disabled = true;
  $('stage').replaceChildren();
  fail('');
  try {
    say('アバターを読み込んでいます…');
    avatar = await create({ container: $('stage'), avatar: `avatars/${AVATAR}/avatar.json`, assetsBaseUrl: ASSETS });
    log('avatar ready', { id: AVATAR, backend: avatar.backend, idleLoops: avatar.idleLoops });
  } catch (e) {
    if (e instanceof WebGPUUnavailableError) {
      fail(`WebGPU が使えないため、アバターを描画できません（${e.reason}）。iPhone は Safari 26 以降、デスクトップは Chrome / Edge 113 以降で開いてください。`);
    } else {
      fail(`アバターを読み込めませんでした: ${String(e && e.message || e)}`);
    }
    log('avatar failed', { error: String(e) });
    placeholder();
    $('start').disabled = false;
    return;
  }
  if (!(await loadTts())) {
    avatar.destroy?.(); avatar = null;
    placeholder();
    $('start').disabled = false;
    return;
  }
  conversation = new Conversation({
    avatar,
    reply,
    synthesize,
    aizuchi: { pick: () => (Math.random() < 0.35 ? 'はい。' : null) },
    emotion: () => ({ angry: 0 }),
  });
  conversation.on('speechstart', (e) => log('speechstart (barge-in)', e));
  conversation.on('speechend', (e) => { log('speechend', e); afterSpeechEnd(e.utt); });
  conversation.on('replystart', (e) => log('replystart', e));
  conversation.on('replysound', (e) => log('replysound', e));
  conversation.on('replydone', (e) => log('replydone', { utt: e.utt }));
  conversation.on('turnend', (e) => log('turnend', e));
  conversation.on('latency', (l) => { log('latency', l); renderStats(); });
  conversation.on('error', (e) => { fail(`返事を作れませんでした: ${e.error.message}`); log('error', { message: e.error.message }); });
  conversation.start();
  for (const id of ['mic', 'stop', 'text', 'send']) $(id).disabled = false;
  say(`準備完了（avatar: ${avatar.backend} / TTS: ${tts.backend}）。「マイク」を押して話しかけてください。`);
}

// --- latency (per turn and medians) ---------------------------------------------------------------------------
function renderStats() {
  if (!conversation) return;
  const s = conversation.stats();
  const m = s.medians;
  const rows = conversation.turnLatencies().slice(-5).map((l) => {
    const b = l.bargeIn ? `  barge-in ${l.bargeIn.stoppedInMs} ms` : '';
    return `#${l.utt}  final ${l.speechEndToFinalMs ?? '-'} / token ${l.speechEndToTokenMs ?? '-'} / sound ${l.speechEndToSoundMs ?? '-'} ms${b}`;
  });
  $('stats').innerHTML = `turns <b>${s.turns}</b>　median speechEnd→final <b>${m.speechEndToFinalMs ?? '-'}</b> ms ／ →first token <b>${m.speechEndToTokenMs ?? '-'}</b> ms ／ →first sound <b>${m.speechEndToSoundMs ?? '-'}</b> ms<br>barge-in ${s.bargeIns}${s.maxBargeInStopMs === null ? '' : `（max stop ${s.maxBargeInStopMs} ms）`}　errors ${s.errors}`
    + (rows.length ? `<br>${rows.join('<br>')}` : '');
}

// --- mic and ASR (without Web Speech API the text box is used as the user's words) ----------------------------
let mic = null;
let asr = null;
let asrText = '';
function startAsr() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const r = new SR();
  r.lang = 'ja-JP';
  r.continuous = true;
  r.interimResults = true;
  r.onresult = (e) => {
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const res = e.results[i];
      if (res.isFinal) asrText += res[0].transcript;
    }
  };
  r.onerror = (e) => log('ASR error', { error: e.error });
  try { r.start(); } catch (e) { log('ASR start failed', { error: String(e) }); return null; }
  return r;
}
/** After the turn end, wait briefly for the ASR final text, then hand it to the conversation (mic turns only). */
function afterSpeechEnd(utt) {
  if (!mic) return;
  const t0 = performance.now();
  void (async () => {
    while (performance.now() - t0 < 800) {
      if (asrText.trim()) break;
      await new Promise((r) => setTimeout(r, 40));
    }
    const heard = asrText.trim();
    asrText = '';
    const text = heard || ($('text').value.trim() !== '' ? $('text').value.trim() : 'こんにちは。');
    log('user final', { utt, text, from: heard ? 'ASR' : 'text box' });
    conversation.userFinal(text);
  })();
}
function micOff() {
  mic?.stop(); mic = null; asr?.stop(); asr = null;
  $('mic').classList.remove('on');
  $('mic').textContent = 'マイク';
}
$('mic').onclick = async () => {
  if (mic) { micOff(); log('mic off'); return; }
  try {
    asrText = '';
    mic = await openMic({ conversation, onFrame: () => undefined });   // the mic first (one permission prompt), then ASR
    asr = startAsr();
    $('mic').classList.add('on');
    $('mic').textContent = 'マイク停止';
    say(asr ? '話しかけてください（ASR: Web Speech API）。' : '話しかけてください（ASR が無いブラウザなので、入力欄の文を発話として使います）。');
    log('mic on', { asr: Boolean(asr) });
  } catch (e) {
    fail(`マイクを使えませんでした: ${String(e && e.message || e)}（https か localhost が必要です）`);
    log('mic failed', { error: String(e) });
  }
};

let booting = false;
$('start').onclick = () => {
  if (booting || conversation) return;   // no double start
  booting = true;
  void boot().finally(() => { booting = false; });
};
$('stop').onclick = () => {
  micOff();
  conversation?.stop();
  conversation = null;
  avatar?.destroy();
  avatar = null;
  placeholder();
  for (const id of ['mic', 'stop', 'text', 'send']) $(id).disabled = true;
  $('start').disabled = false;
  say('停止しました。');
  log('stop');
};
const send = () => {
  const text = $('text').value.trim();
  if (!text || !conversation) return;
  log('sendText', { text });
  conversation.sendText(text);
  $('text').value = '';
};
$('send').onclick = send;
$('text').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.isComposing) send(); });

// --- settings (a change reloads with the query) ---------------------------------------------------------------
function fillSelect(sel, items, current, param) {
  for (const it of items) {
    const o = document.createElement('option');
    o.value = it.id; o.textContent = it.name; o.selected = it.id === current;
    sel.append(o);
  }
  sel.onchange = () => { const u = new URL(location.href); u.searchParams.set(param, sel.value); location.href = u.href; };
}
fillSelect($('avatarSel'), AVATAR_LIST, AVATAR, 'avatar');
fillSelect($('voiceSel'), VOICE_LIST, VOICE, 'voice');
fillSelect($('backendSel'), [{ id: 'auto', name: 'auto' }, { id: 'webgpu', name: 'webgpu' }, { id: 'cpu', name: 'cpu (WebAssembly)' }], BACKEND, 'ttsbackend');
if (!AVATARS.length) { fail(`アバターが見つかりません（${ASSETS}avatars/）`); $('start').disabled = true; }
log('page', { sdk: SDK, assets: ASSETS, avatar: AVATAR, voice: VOICE, ttsbackend: BACKEND });

// handle for headless checks
window.zenConv = {
  get conversation() { return conversation; },
  get avatar() { return avatar; },
  get tts() { return tts; },
  boot,
  sendText: (t) => conversation.sendText(t),
  stats: () => conversation?.stats() ?? null,
  latencies: () => conversation?.turnLatencies() ?? [],
};
