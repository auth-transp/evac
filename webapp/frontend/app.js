// Live Service: λογική της εφαρμογής (βήματα, εργαλεία, επικοινωνία με το backend)
import { post, toB64, fromB64 } from './api.js';
import { Viewport } from './view.js';
import { render } from './render.js';
import { mapAvailable, showMap, startDraw, clearPolygon, getPolygon, getMap, onPolygonChange, searchPlaces, goTo } from './map.js';
import { sim, initSim, startSim, cancelSim, resetSim, togglePlay, userScrub, setSpeed, drawCharts } from './sim.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas');

// Κλάσεις: 0 εκτός, 1 έδαφος, 2 δρόμος, 3 κτίριο, 4 τοίχος· χρώματα overlay ως RGBA (little-endian Uint32)
const pack = (r, g, b, a) => ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
const PALETTE = [pack(0, 0, 0, 150), 0, pack(255, 255, 255, 200), pack(233, 144, 42, 150), pack(30, 30, 30, 240)];
const TOOL_NAMES = { none: 'Μετακίνηση', region: 'Περιοχή', brush: 'Πινέλο', exit: 'Έξοδος', spawn: 'Ζώνη spawn' };

// Το project προέρχεται από δεδομένα OpenStreetMap: το grid (m/pixel srcMpp) είναι η «εικόνα» του καμβά
const s = {
  id: null, img: null, imgW: 0, imgH: 0,
  srcMpp: null,
  region: [], spawn: [], exits: [], draft: null,
  grid: null, labels: null, labelsCanvas: null, labelsCtx: null, labelsData: null, labelsPix: null, history: [],
  hmImg: null, built: false, showCm: true, showSpawn: true,
  view: 'image', tool: 'none', step: 'image', regionMode: 'add',
  brushClass: 1, brushSize: 3, brushCursor: null,
  hover: null,                                     // στοιχείο λίστας πάνω από το οποίο βρίσκεται το ποντίκι: {key: 'region' | 'exits' | 'spawn', i}
};

const vp = new Viewport(canvas, redraw);
function redraw() {
  render(canvas, vp, s);
  $('hud').textContent = `${TOOL_NAMES[s.tool]} · ${Math.round(vp.scale * 100)}%`;
}

const near = (a, b, tol = 10) => Math.hypot(a[0] - b[0], a[1] - b[1]) * vp.scale < tol;    // εγγύτητα σε css pixels

// ---------- βήματα και εργαλεία ----------
const STEP_TOOL = { image: () => 'none', region: () => 'region', analyze: () => (s.labels ? 'brush' : 'none'), exits: () => 'exit', spawn: () => 'spawn', build: () => 'none', sim: () => 'none' };

function setStep(name) {
  s.step = name;
  if (name === 'sim') requestAnimationFrame(drawCharts);       // τα γραφήματα μετρούν το πλάτος τους όταν το βήμα είναι ορατό
  document.querySelectorAll('.step').forEach(el => el.classList.toggle('active', el.dataset.step === name));
  updateMapVisibility();
  setTool(STEP_TOOL[name]());
}

// Ο χάρτης OpenStreetMap εμφανίζεται στη θέση του καμβά μόνο στο βήμα 1
function updateMapVisibility() {
  const want = s.step === 'image';
  showMap(want && mapAvailable());
  if (want && !mapAvailable()) setInfo('mapInfo', 'Ο χάρτης δεν φόρτωσε (χρειάζεται σύνδεση στο Internet για τη βιβλιοθήκη Leaflet).');
}

function setTool(t) {
  s.draft = null;
  s.tool = t;
  vp.tool = TOOLS[t] || null;
  if (t === 'brush' && s.view !== 'labels') setView('labels');
  [['brushTool', 'brush'], ['exitTool', 'exit'], ['spawnTool', 'spawn']]
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
  exit: dragTool(
    p => { s.draft = { kind: 'exit', points: [p], cursor: p }; },
    p => { s.draft.cursor = p; },
    p => {
      const a = s.draft.points[0];
      s.draft = null;
      if (!s.srcMpp) return;                                          // χωρίς εισαγωγή από τον χάρτη δεν υπάρχει κλίμακα
      if (Math.hypot(p[0] - a[0], p[1] - a[1]) * s.srcMpp >= 2) s.exits.push({ p1: a, p2: p, thickness_m: +$('exitThickness').value });
      refreshLists();
    }),
  brush: {
    down(p) { if (!s.labels) return; pushHistory(); dragging = true; s.lastCell = null; paintAt(p); },
    move(p, e, pressed) { s.brushCursor = p; if (dragging && pressed) paintAt(p); else redraw(); },
    up() { dragging = false; },
  },
};

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

// Δίσκος διαμέτρου s.brushSize cells γύρω από το cell (cx, cy): μέγεθος 1 = ένα cell, 2 = σταυρός 5 cells, 3 = 3×3, κ.ο.κ.
function paintDisc(cx, cy) {
  const { w, h } = s.grid, rad = s.brushSize / 2, ext = Math.ceil(rad);
  for (let y = Math.max(0, cy - ext); y <= Math.min(h - 1, cy + ext); y++)
    for (let x = Math.max(0, cx - ext); x <= Math.min(w - 1, cx + ext); x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= rad * rad) { const i = y * w + x; s.labels[i] = s.brushClass; s.labelsPix[i] = PALETTE[s.brushClass]; }
}

function paintAt(p) {
  const g = s.grid;
  const cx = Math.floor((p[0] - g.x0) * g.f), cy = Math.floor((p[1] - g.y0) * g.f);
  s.brushCursor = p;
  const last = s.lastCell || [cx, cy], steps = Math.max(1, Math.ceil(Math.hypot(cx - last[0], cy - last[1]) / Math.max(1, s.brushSize / 4)));
  for (let k = 1; k <= steps; k++) paintDisc(Math.round(last[0] + (cx - last[0]) * k / steps), Math.round(last[1] + (cy - last[1]) * k / steps));
  s.lastCell = [cx, cy];
  s.labelsCtx.putImageData(s.labelsData, 0, 0);
  s.built = false;
  redraw();
}

// ---------- επικοινωνία με το backend ----------
// kind: 'error' (κόκκινο, role=alert), 'loading' (spinner) ή '' (απλό μήνυμα)
function setInfo(id, text, kind = '') {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle('error', kind === 'error');
  el.classList.toggle('loading', kind === 'loading');
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
}
const setError = (id, text) => setInfo(id, text, 'error');

// Κατάσταση «απασχολημένο» για κουμπί: spinner και απενεργοποίηση όσο τρέχει το αίτημα
function setBusy(btn, on) {
  btn.classList.toggle('busy', on);
  btn.setAttribute('aria-busy', on);
  btn.disabled = on;
}

// Έλεγχος αριθμητικού πεδίου: κόκκινο περίγραμμα και μήνυμα στο infoId όταν η τιμή δεν είναι έγκυρη
function checkNumber(inputId, infoId, label, min, max) {
  const el = $(inputId), v = el.valueAsNumber;
  const ok = Number.isFinite(v) && v >= min && v <= max;
  el.classList.toggle('invalid', !ok);
  el.setAttribute('aria-invalid', !ok);
  if (!ok) { setError(infoId, `${label}: δώσε αριθμό από ${min} έως ${max}.`); el.focus(); }
  return ok;
}
['nAgents', 'exitThickness', 'simT', 'simSeed'].forEach(id => $(id).addEventListener('input', e => { e.target.classList.remove('invalid'); e.target.removeAttribute('aria-invalid'); }));

// Εισαγωγή από OpenStreetMap: το backend επιστρέφει έτοιμο χάρτη κλάσεων, περίγραμμα και προτάσεις εξόδων για το πολύγωνο του χάρτη
async function importOsm() {
  const poly = getPolygon();
  if (!poly) return setError('mapInfo', 'Σχεδίασε πρώτα την περιοχή στον χάρτη.');
  setBusy($('mapImport'), true);
  setInfo('mapInfo', 'Λήψη δεδομένων από το OpenStreetMap… (αν οι server είναι φορτωμένοι μπορεί να πάρει λίγα λεπτά)', 'loading');
  try {
    const r = await post('/api/osm', { polygon: poly });                         // η ανάλυση του grid επιλέγεται αυτόματα από το μέγεθος της περιοχής
    resetSim();
    const g = r.grid, base = document.createElement('canvas');
    base.width = g.w; base.height = g.h;
    Object.assign(s, { id: r.id, img: base, imgW: g.w, imgH: g.h, srcMpp: g.mpp,        // η κλίμακα (m/pixel) προκύπτει από τις γεωγραφικές συντεταγμένες
      region: [{ mode: 'add', points: r.region }], spawn: [], exits: r.exits.map(e => ({ p1: e.p1, p2: e.p2, thickness_m: e.thickness_m })),
      hmImg: null, built: false, view: 'labels' });
    const labels = fromB64(r.labels);
    initLabels(g, labels);
    paintBase(base, labels);
    refreshLists();
    vp.fit(g.w, g.h);
    const c = r.counts;
    const msg = `Βρέθηκαν ${c.buildings} κτίρια, ${c.roads} δρόμοι, ${c.gates} πύλες· προτάθηκαν ${r.exits.length} έξοδοι. ` +
      `Πόρτες κτιρίων: ${c.doors_osm} από εισόδους του OSM, ${c.doors_auto} αυτόματες (χρώμα κτιρίου στον τοίχο· ξαναβάψε με το πινέλο «Κτίριο» ή «Τοίχος» για αλλαγές).` + (r.warnings.length ? ' ' + r.warnings.join(' ') : '');
    setInfo('mapInfo', msg);
    setInfo('correctionInfo', `1 pixel = ${g.mpp.toFixed(2)} m · η περιοχή καλύπτει ${(g.w * g.mpp).toFixed(0)} × ${(g.h * g.mpp).toFixed(0)} m. ` + msg);
    setView('labels');
    setStep('analyze');
  } catch (e) {
    setError('mapInfo', 'Σφάλμα: ' + (e.message || e));
  } finally {
    setBusy($('mapImport'), false);
    $('mapImport').disabled = !getPolygon();
  }
}

// Βασική «εικόνα» του project από τον χάρτη κλάσεων: έδαφος, δρόμοι, κτίρια, τοίχοι
function paintBase(canvas, labels) {
  const x = canvas.getContext('2d'), d = x.createImageData(canvas.width, canvas.height), px = new Uint32Array(d.data.buffer);
  const col = [pack(27, 29, 34, 255), pack(205, 182, 138, 255), pack(232, 232, 232, 255), pack(176, 120, 74, 255), pack(58, 42, 28, 255)];
  for (let i = 0; i < labels.length; i++) px[i] = col[labels[i]] ?? 0;
  x.putImageData(d, 0, 0);
}

async function runBuild() {
  if (!s.labels) return setError('buildInfo', 'Εισήγαγε πρώτα δεδομένα από το OpenStreetMap (βήμα 1).');
  if (!checkNumber('nAgents', 'buildInfo', 'Πλήθος agents', 1, 100000)) return;
  setBusy($('build'), true);
  setInfo('buildInfo', 'Δημιουργία…', 'loading');
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
    setError('buildInfo', 'Σφάλμα: ' + (e.message || e));
  } finally {
    setBusy($('build'), false);
  }
}

// ---------- λίστες και ενδείξεις ----------
function listHtml(items, fmt) { return items.map((it, i) => `<li data-i="${i}"><span>${fmt(it, i)}</span><button data-i="${i}" title="Διαγραφή">✕</button></li>`).join(''); }

function refreshLists() {
  s.hover = null;                                   // οι δείκτες των λιστών αλλάζουν όταν προστίθεται/διαγράφεται στοιχείο
  $('regionList').innerHTML = listHtml(s.region, (r, i) => `Περιοχή ${i + 1} · ${r.mode === 'add' ? 'προσθήκη' : 'αφαίρεση'} · ${r.points.length} κορυφές`);
  $('exitList').innerHTML = listHtml(s.exits, (e, i) => `Έξοδος ${i + 1} · ${(Math.hypot(e.p2[0] - e.p1[0], e.p2[1] - e.p1[1]) * (s.srcMpp || 0)).toFixed(0)} m × ${e.thickness_m} m`);
  $('spawnList').innerHTML = listHtml(s.spawn, (z, i) => `Ζώνη ${i + 1} · ${z.points.length} κορυφές`);
  markDone();
}

function markDone() {
  const done = { image: !!s.img, region: s.region.length > 0, analyze: !!s.labels, exits: s.exits.length > 0, spawn: s.spawn.length > 0, build: s.built, sim: sim.status === 'done' };
  document.querySelectorAll('.step').forEach(el => el.classList.toggle('done', !!done[el.dataset.step]));
}

// ---------- σύνδεση UI ----------
document.querySelectorAll('.step h2').forEach(h => {
  h.tabIndex = 0;                                                              // προσβάσιμο και από πληκτρολόγιο
  h.setAttribute('role', 'button');
  h.addEventListener('click', () => setStep(h.parentElement.dataset.step));
  h.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setStep(h.parentElement.dataset.step); } });
});
$('mapSearch').addEventListener('input', e => e.target.classList.remove('invalid'));
document.querySelectorAll('#viewToggle button').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
$('regionMode').querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
  s.regionMode = b.dataset.mode;
  $('regionMode').querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
  if (s.draft?.kind === 'region') s.draft.mode = s.regionMode;
  setTool('region');
}));

$('brushSize').addEventListener('input', e => { s.brushSize = +e.target.value; $('brushOut').textContent = e.target.value; });
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

// Hover πάνω σε στοιχείο λίστας (περιοχή, έξοδος, ζώνη spawn): φωτίζεται στον χάρτη με έντονο ματζέντα χρώμα (render.js)
[['regionList', 'region'], ['exitList', 'exits'], ['spawnList', 'spawn']].forEach(([id, key]) => {
  const ul = $(id);
  ul.addEventListener('mouseover', e => {
    const li = e.target.closest('li');
    if (!li || li.dataset.i === undefined) return;
    s.hover = { key, i: +li.dataset.i };
    redraw();
  });
  ul.addEventListener('mouseleave', () => { s.hover = null; redraw(); });
});

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
  if (!s.id || !s.built) return setError('simInfo', 'Δημιούργησε πρώτα το heightmap (βήμα 6) και μην το αλλάξεις μετά.');
  if (!checkNumber('simT', 'simInfo', 'Διάρκεια', 10, 20000) || !checkNumber('simSeed', 'simInfo', 'Seed', 0, 2147483647)) return;
  if (s.view === 'labels') setView('heightmap');
  startSim(s.id, +$('simT').value, +$('simSeed').value);
});
$('simCancel').addEventListener('click', cancelSim);
$('simPlay').addEventListener('click', togglePlay);
$('simSlider').addEventListener('input', e => userScrub(+e.target.value));
$('simSpeed').addEventListener('change', e => setSpeed(+e.target.value));
$('showCm').addEventListener('change', e => { s.showCm = e.target.checked; redraw(); });
// εμφάνιση/απόκρυψη ζωνών spawn: ο διακόπτης της μπάρας animation και ο διακόπτης της καρτέλας Προσομοίωση είναι συγχρονισμένοι
['showSpawn', 'showSpawnBar'].forEach(id => $(id).addEventListener('change', e => {
  s.showSpawn = e.target.checked;
  $('showSpawn').checked = $('showSpawnBar').checked = s.showSpawn;
  redraw();
}));

// ---------- χάρτης OpenStreetMap ----------
$('mapDraw').addEventListener('click', startDraw);
$('mapClear').addEventListener('click', clearPolygon);
$('mapImport').addEventListener('click', importOsm);
onPolygonChange(poly => { $('mapImport').disabled = !poly; });
$('mapSearchForm').addEventListener('submit', async e => {
  e.preventDefault();
  const q = $('mapSearch').value.trim();
  const list = $('mapResults');
  $('mapSearch').classList.toggle('invalid', !q);
  if (!q) { list.innerHTML = '<li class="error" role="alert">Γράψε έναν τόπο για αναζήτηση.</li>'; return; }
  list.innerHTML = '<li class="loading">Αναζήτηση…</li>';
  try {
    const places = await searchPlaces(q);
    list.innerHTML = places.length ? '' : '<li>Δεν βρέθηκε τόπος.</li>';
    places.forEach(p => {
      const li = document.createElement('li');
      li.textContent = p.name;
      li.addEventListener('click', () => { goTo(p); list.innerHTML = ''; });
      list.appendChild(li);
    });
  } catch (err) {
    const li = document.createElement('li');
    li.className = 'error';
    li.setAttribute('role', 'alert');
    li.textContent = 'Σφάλμα αναζήτησης: ' + (err.message || err);          // textContent: το μήνυμα δεν ερμηνεύεται ως HTML
    list.replaceChildren(li);
  }
});

window.liveService = { s, vp, sim, getMap };       // για έλεγχο/αποσφαλμάτωση από την κονσόλα του browser
refreshLists();
setStep('image');
