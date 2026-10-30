// Live Service: λογική της εφαρμογής (βήματα, εργαλεία, επικοινωνία με το backend)
import { upload, post, toB64, fromB64 } from './api.js';
import { Viewport } from './view.js';
import { render } from './render.js';
import { sim, initSim, startSim, cancelSim, resetSim, togglePlay, userScrub, setSpeed, drawCharts } from './sim.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas');

// Κλάσεις: 0 εκτός, 1 έδαφος, 2 δρόμος, 3 κτίριο, 4 τοίχος· χρώματα overlay ως RGBA (little-endian Uint32)
const pack = (r, g, b, a) => ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
const PALETTE = [pack(0, 0, 0, 150), 0, pack(255, 255, 255, 200), pack(233, 144, 42, 150), pack(30, 30, 30, 240)];
const TOOL_NAMES = { none: 'Μετακίνηση', ref: 'Γραμμή αναφοράς', region: 'Περιοχή', sample: 'Δείγμα εδάφους', brush: 'Πινέλο', exit: 'Έξοδος', spawn: 'Ζώνη spawn' };

const s = {
  id: null, img: null, imgW: 0, imgH: 0,
  srcMpp: null, refLine: null,
  region: [], spawn: [], exits: [], draft: null,
  grid: null, labels: null, labelsCanvas: null, labelsCtx: null, labelsData: null, labelsPix: null, history: [],
  hmImg: null, built: false, groundSample: null, showCm: true,
  view: 'image', tool: 'none', step: 'image', regionMode: 'add',
  brushClass: 1, brushSize: 3, brushCursor: null,
};

const vp = new Viewport(canvas, redraw);
function redraw() {
  render(canvas, vp, s);
  $('hud').textContent = `${TOOL_NAMES[s.tool]} · ${Math.round(vp.scale * 100)}%`;
}

const near = (a, b, tol = 10) => Math.hypot(a[0] - b[0], a[1] - b[1]) * vp.scale < tol;    // εγγύτητα σε css pixels

// ---------- βήματα και εργαλεία ----------
const STEP_TOOL = { image: () => 'none', scale: () => ($('scaleLine').classList.contains('hidden') ? 'none' : 'ref'), region: () => 'region', analyze: () => (s.labels ? 'brush' : 'none'), exits: () => 'exit', spawn: () => 'spawn', build: () => 'none', sim: () => 'none' };

function setStep(name) {
  s.step = name;
  if (name === 'sim') requestAnimationFrame(drawCharts);       // τα γραφήματα μετρούν το πλάτος τους όταν το βήμα είναι ορατό
  document.querySelectorAll('.step').forEach(el => el.classList.toggle('active', el.dataset.step === name));
  setTool(STEP_TOOL[name]());
}

function setTool(t) {
  s.draft = null;
  s.tool = t;
  vp.tool = TOOLS[t] || null;
  if (t === 'brush' && s.view !== 'labels') setView('labels');
  [['drawRef', 'ref'], ['brushTool', 'brush'], ['exitTool', 'exit'], ['spawnTool', 'spawn'], ['pickGround', 'sample']]
    .forEach(([id, tool]) => $(id).classList.toggle('active', t === tool));
  canvas.style.cursor = t === 'none' ? 'grab' : 'crosshair';
  redraw();
}

function setView(v) {
  if ((v === 'labels' && !s.labels) || (v === 'heightmap' && !s.hmImg)) return;
  s.view = v;
  document.querySelectorAll('#viewToggle button').forEach(b => b.classList.toggle('active', b.dataset.view === v));
  redraw();
}

function finishPoly() {
  const d = s.draft;
  if (d && d.points.length >= 3) {
    if (d.kind === 'region') s.region.push({ mode: d.mode, points: d.points });
    else s.spawn.push({ points: d.points });
  }
  s.draft = null;
  refreshLists();
  redraw();
}

const polyTool = kind => ({
  click(p) {
    if (!s.draft || s.draft.kind !== kind) s.draft = { kind, mode: s.regionMode, points: [], cursor: null };
    const pts = s.draft.points;
    if (pts.length >= 3 && near(p, pts[0])) return finishPoly();
    if (pts.length && near(p, pts[pts.length - 1], 3)) return;      // αγνοεί το διπλό κλικ του dblclick
    pts.push(p);
    redraw();
  },
  move(p) { if (s.draft?.kind === kind) { s.draft.cursor = p; redraw(); } },
  dblclick() { finishPoly(); },
});

let dragging = false;
const dragTool = (onStart, onMove, onEnd) => ({
  down(p) { dragging = true; onStart(p); redraw(); },
  move(p, e, pressed) { if (dragging && pressed) { onMove(p); redraw(); } },
  up(p) { if (dragging) { dragging = false; onEnd(p); redraw(); } },
});

const TOOLS = {
  none: null,
  region: polyTool('region'),
  spawn: polyTool('spawn'),
  ref: dragTool(p => { s.refLine = [p, p]; }, p => { s.refLine[1] = p; }, () => { updateScale(); }),
  exit: dragTool(
    p => { s.draft = { kind: 'exit', points: [p], cursor: p }; },
    p => { s.draft.cursor = p; },
    p => {
      const a = s.draft.points[0];
      s.draft = null;
      if (!s.srcMpp) return setInfo('scaleInfo', 'Όρισε πρώτα την κλίμακα (βήμα 2).');
      if (Math.hypot(p[0] - a[0], p[1] - a[1]) * s.srcMpp >= 2) s.exits.push({ p1: a, p2: p, thickness_m: +$('exitThickness').value });
      refreshLists();
    }),
  sample: { click(p) { s.groundSample = p; setInfo('proposeInfo', 'Ορίστηκε δείγμα εδάφους.'); setTool('none'); } },
  brush: {
    down(p) { if (!s.labels) return; pushHistory(); dragging = true; s.lastCell = null; paintAt(p); },
    move(p, e, pressed) { s.brushCursor = p; if (dragging && pressed) paintAt(p); else redraw(); },
    up() { dragging = false; },
  },
};

// ---------- κλίμακα ----------
function updateScale() {
  const mode = document.querySelector('input[name=scaleMode]:checked').value;
  let mpp = null;
  if (mode === 'line') {
    const m = +$('refMeters').value;
    if (s.refLine && m > 0) { const len = Math.hypot(s.refLine[1][0] - s.refLine[0][0], s.refLine[1][1] - s.refLine[0][1]); if (len > 2) mpp = m / len; }
  } else {
    const lat = +$('geoLat').value, zoom = +$('geoZoom').value;
    if (isFinite(lat) && isFinite(zoom)) mpp = 156543.03392 * Math.cos(lat * Math.PI / 180) / 2 ** zoom;
  }
  s.srcMpp = mpp;
  setInfo('scaleInfo', mpp ? `1 pixel εικόνας = ${mpp.toFixed(3)} m · η εικόνα καλύπτει ${(s.imgW * mpp).toFixed(0)} × ${(s.imgH * mpp).toFixed(0)} m` : 'Η κλίμακα δεν έχει οριστεί.');
  refreshLists();
  redraw();
}

// ---------- χάρτης κλάσεων και πινέλο ----------
function initLabels(grid, u8) {
  s.grid = grid; s.labels = u8;
  s.labelsCanvas = document.createElement('canvas');
  s.labelsCanvas.width = grid.w; s.labelsCanvas.height = grid.h;
  s.labelsCtx = s.labelsCanvas.getContext('2d');
  s.labelsData = s.labelsCtx.createImageData(grid.w, grid.h);
  s.labelsPix = new Uint32Array(s.labelsData.data.buffer);
  s.history = [];
  refreshLabels();
}

function refreshLabels() {
  for (let i = 0; i < s.labels.length; i++) s.labelsPix[i] = PALETTE[s.labels[i]] ?? 0;
  s.labelsCtx.putImageData(s.labelsData, 0, 0);
}

function pushHistory() { s.history.push(s.labels.slice()); if (s.history.length > 30) s.history.shift(); }

function paintDisc(cx, cy) {
  const { w, h } = s.grid, r = s.brushSize;
  for (let y = Math.max(0, cy - r); y <= Math.min(h - 1, cy + r); y++)
    for (let x = Math.max(0, cx - r); x <= Math.min(w - 1, cx + r); x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) { const i = y * w + x; s.labels[i] = s.brushClass; s.labelsPix[i] = PALETTE[s.brushClass]; }
}

function paintAt(p) {
  const g = s.grid;
  const cx = Math.floor((p[0] - g.x0) * g.f), cy = Math.floor((p[1] - g.y0) * g.f);
  s.brushCursor = p;
  const last = s.lastCell || [cx, cy], steps = Math.max(1, Math.ceil(Math.hypot(cx - last[0], cy - last[1]) / Math.max(1, s.brushSize / 2)));
  for (let k = 1; k <= steps; k++) paintDisc(Math.round(last[0] + (cx - last[0]) * k / steps), Math.round(last[1] + (cy - last[1]) * k / steps));
  s.lastCell = [cx, cy];
  s.labelsCtx.putImageData(s.labelsData, 0, 0);
  s.built = false;
  redraw();
}

// ---------- επικοινωνία με το backend ----------
function setInfo(id, text) { $(id).textContent = text; }

async function loadFile(file) {
  if (!file) return;
  setInfo('imageInfo', 'Ανέβασμα…');
  try {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.src = url;
    await img.decode();
    const up = await upload(file);
    if (up.width !== img.naturalWidth || up.height !== img.naturalHeight) throw new Error('Οι διαστάσεις της εικόνας διαφέρουν από αυτές του server (EXIF περιστροφή;)');
    resetSim();
    Object.assign(s, { id: up.id, img, imgW: up.width, imgH: up.height, refLine: null, srcMpp: null, region: [], spawn: [], exits: [], grid: null, labels: null, hmImg: null, built: false, groundSample: null, view: 'image' });
    setInfo('imageInfo', `${file.name} · ${up.width} × ${up.height} px`);
    updateScale();
    vp.fit(up.width, up.height);
    setStep('scale');
  } catch (e) {
    setInfo('imageInfo', 'Σφάλμα: ' + (e.message || e));
  }
}

async function runPropose() {
  if (!s.id) return setInfo('proposeInfo', 'Ανέβασε πρώτα εικόνα.');
  if (!s.srcMpp) return setInfo('proposeInfo', 'Όρισε πρώτα την κλίμακα (βήμα 2).');
  if (!s.region.some(r => r.mode === 'add')) return setInfo('proposeInfo', 'Όρισε πρώτα την περιοχή εγκατάστασης (βήμα 3).');
  setInfo('proposeInfo', 'Επεξεργασία…');
  try {
    const r = await post('/api/propose', {
      id: s.id, src_mpp: s.srcMpp, target_mpp: +$('targetMpp').value, region: s.region,
      sensitivity: +$('sensitivity').value, min_building_m: +$('minBuilding').value, ground_sample: s.groundSample,
    });
    if (s.labels) pushHistory();
    const hist = s.history;
    initLabels(r.grid, fromB64(r.labels));
    s.history = hist.length && hist[0].length === s.labels.length ? hist : [];
    s.built = false;
    setInfo('proposeInfo', `Grid ${r.grid.w} × ${r.grid.h} · ${r.grid.mpp.toFixed(2)} m/pixel. Διόρθωσε με το πινέλο όπου χρειάζεται.`);
    setView('labels');
    setTool('brush');
    markDone();
  } catch (e) {
    setInfo('proposeInfo', 'Σφάλμα: ' + (e.message || e));
  }
}

async function runBuild() {
  if (!s.labels) return setInfo('buildInfo', 'Τρέξε πρώτα την αυτόματη πρόταση (βήμα 4).');
  setInfo('buildInfo', 'Δημιουργία…');
  try {
    const r = await post('/api/build', { id: s.id, grid: s.grid, labels: toB64(s.labels), n_agents: +$('nAgents').value, exits: s.exits, spawn_zones: s.spawn });
    const base = `/api/files/${s.id}`;
    $('dlHeightmap').href = `${base}/heightmap.png`; $('dlSpawn').href = `${base}/spawn_mask.png`; $('dlConfig').href = `${base}/config.json`;
    $('downloads').classList.remove('hidden');
    $('warnList').innerHTML = r.config.warnings.map(w => `<li>${w}</li>`).join('');
    setInfo('buildInfo', `Heightmap ${r.config.cols} × ${r.config.rows} · ${r.config.meters_per_pixel.toFixed(2)} m/pixel · ${r.walkable_cells} βατά cells · ${r.config.exits.length} έξοδοι · ${r.config.spawn_zones.length} ζώνες spawn`);
    const img = new Image();
    img.onload = () => { s.hmImg = img; s.built = true; setView('heightmap'); markDone(); };
    img.src = `${base}/heightmap.png?t=${Date.now()}`;
  } catch (e) {
    setInfo('buildInfo', 'Σφάλμα: ' + (e.message || e));
  }
}

// ---------- λίστες και ενδείξεις ----------
function listHtml(items, fmt) { return items.map((it, i) => `<li><span>${fmt(it, i)}</span><button data-i="${i}" title="Διαγραφή">✕</button></li>`).join(''); }

function refreshLists() {
  $('regionList').innerHTML = listHtml(s.region, (r, i) => `Περιοχή ${i + 1} · ${r.mode === 'add' ? 'προσθήκη' : 'αφαίρεση'} · ${r.points.length} κορυφές`);
  $('exitList').innerHTML = listHtml(s.exits, (e, i) => `Έξοδος ${i + 1} · ${(Math.hypot(e.p2[0] - e.p1[0], e.p2[1] - e.p1[1]) * (s.srcMpp || 0)).toFixed(0)} m × ${e.thickness_m} m`);
  $('spawnList').innerHTML = listHtml(s.spawn, (z, i) => `Ζώνη ${i + 1} · ${z.points.length} κορυφές`);
  markDone();
}

function markDone() {
  const done = { image: !!s.img, scale: !!s.srcMpp, region: s.region.length > 0, analyze: !!s.labels, exits: s.exits.length > 0, spawn: s.spawn.length > 0, build: s.built, sim: sim.status === 'done' };
  document.querySelectorAll('.step').forEach(el => el.classList.toggle('done', !!done[el.dataset.step]));
}

// ---------- σύνδεση UI ----------
document.querySelectorAll('.step h2').forEach(h => h.addEventListener('click', () => setStep(h.parentElement.dataset.step)));
document.querySelectorAll('#viewToggle button').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
$('file').addEventListener('change', e => loadFile(e.target.files[0]));
const drop = $('drop');
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => loadFile(e.dataTransfer.files[0]));

document.querySelectorAll('input[name=scaleMode]').forEach(r => r.addEventListener('change', () => {
  const line = r.value === 'line' && r.checked;
  if (r.checked) { $('scaleLine').classList.toggle('hidden', !line); $('scaleGeo').classList.toggle('hidden', line); }
  updateScale();
  setTool(STEP_TOOL.scale());
}));
['refMeters', 'geoLat', 'geoZoom'].forEach(id => $(id).addEventListener('input', updateScale));
$('drawRef').addEventListener('click', () => setTool('ref'));

$('regionMode').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
  s.regionMode = b.dataset.mode;
  $('regionMode').querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
  if (s.draft?.kind === 'region') s.draft.mode = s.regionMode;
  setTool('region');
}));

$('sensitivity').addEventListener('input', e => { $('sensitivityOut').textContent = e.target.value; });
$('brushSize').addEventListener('input', e => { s.brushSize = +e.target.value; $('brushOut').textContent = e.target.value; });
$('pickGround').addEventListener('click', () => setTool('sample'));
$('propose').addEventListener('click', runPropose);
$('palette').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
  s.brushClass = +b.dataset.class;
  $('palette').querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
  if (s.labels) setTool('brush');
}));
$('brushTool').addEventListener('click', () => s.labels && setTool('brush'));
$('undoLabels').addEventListener('click', () => { if (s.history.length) { s.labels.set(s.history.pop()); refreshLabels(); redraw(); } });
$('exitTool').addEventListener('click', () => setTool('exit'));
$('spawnTool').addEventListener('click', () => setTool('spawn'));
$('build').addEventListener('click', runBuild);

[['regionList', 'region'], ['exitList', 'exits'], ['spawnList', 'spawn']].forEach(([id, key]) => $(id).addEventListener('click', e => {
  const i = e.target.dataset?.i;
  if (i === undefined) return;
  s[key].splice(+i, 1);
  s.built = false;
  refreshLists();
  redraw();
}));

window.addEventListener('keydown', e => {
  if (/INPUT|TEXTAREA/.test(document.activeElement.tagName) && document.activeElement.type !== 'range') return;
  if (e.key === 'Enter' && s.draft) finishPoly();
  else if (e.key === 'Escape') { s.draft = null; dragging = false; redraw(); }
  else if (e.key === 'Backspace' && s.draft?.points.length) { s.draft.points.pop(); redraw(); }
});

// ---------- προσομοίωση ----------
initSim(() => { redraw(); markDone(); });
$('simRun').addEventListener('click', () => {
  if (!s.id || !s.built) return setInfo('simInfo', 'Δημιούργησε πρώτα το heightmap (βήμα 7) και μην το αλλάξεις μετά.');
  if (s.view === 'labels') setView('heightmap');
  startSim(s.id, +$('simT').value, +$('simSeed').value);
});
$('simCancel').addEventListener('click', cancelSim);
$('simPlay').addEventListener('click', togglePlay);
$('simSlider').addEventListener('input', e => userScrub(+e.target.value));
$('simSpeed').addEventListener('change', e => setSpeed(+e.target.value));
$('showCm').addEventListener('change', e => { s.showCm = e.target.checked; redraw(); });

window.liveService = { s, vp, sim };       // για έλεγχο/αποσφαλμάτωση από την κονσόλα του browser
refreshLists();
setStep('image');
