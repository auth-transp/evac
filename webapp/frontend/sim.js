// Προσομοίωση στον browser: εκκίνηση/ακύρωση job, πρόοδος, ζωντανή αναπαραγωγή των frames και γραφήματα.
// Το frames.bin έχει ανά frame και agent 4 Float32: στήλη, γραμμή (συντεταγμένες grid), τοξικό φορτίο, διέφυγε (0/1).
import { post } from './api.js';

const $ = id => document.getElementById(id);
const FPS = 30;                                         // ταχύτητα αναπαραγωγής 1× (frames ανά δευτερόλεπτο, όπως τα βίντεο των σεναρίων)

export const sim = {
  id: null, meta: null, data: null, n: 0, stride: 4, nFrames: 0, frame: 0,
  status: 'none', T: 0, playing: false, follow: true, speed: 1,
  cm: [], esc: [], dead: [], timer: null, lastTick: 0, nextFetch: 0,
};

let redraw = () => {};
export function initSim(redrawFn) { redraw = redrawFn; }

const getJson = async url => (await fetch(url + (url.includes('?') ? '&' : '?') + 't=' + Date.now())).json();
const fileUrl = (id, name) => `/api/files/${id}/${name}`;

export function resetSim() {
  clearInterval(sim.timer);
  Object.assign(sim, { id: null, meta: null, data: null, n: 0, nFrames: 0, frame: 0, status: 'none', playing: false, follow: true, cm: [], esc: [], dead: [], timer: null });
  $('simbar').classList.add('hidden');
  $('simDownloads').classList.add('hidden');
  $('simProgress').classList.add('hidden');
  $('simCancel').classList.add('hidden');
  $('simRun').disabled = false;
  $('simInfo').textContent = '';
  drawCharts();
}

// ---------- job ----------
export async function startSim(id, T, seed) {
  resetSim();
  sim.id = id; sim.T = T; sim.status = 'starting';
  $('simInfo').textContent = 'Εκκίνηση της διεργασίας Julia… (η πρώτη φορά αργεί λίγο λόγω φόρτωσης)';
  try {
    await post('/api/run', { id, T, seed });
  } catch (e) {
    sim.status = 'failed';
    $('simInfo').textContent = 'Σφάλμα: ' + (e.message || e);
    return;
  }
  $('simRun').disabled = true;
  $('simCancel').classList.remove('hidden');
  $('simProgress').classList.remove('hidden');
  sim.timer = setInterval(poll, 1000);
  poll();
}

export async function cancelSim() {
  if (!sim.id) return;
  await post(`/api/jobs/${sim.id}/cancel`, {});
}

async function poll() {
  const id = sim.id;
  if (!id) return;
  let st;
  try { st = await getJson(`/api/jobs/${id}`); } catch { return; }
  if (id !== sim.id) return;
  sim.status = st.status;
  $('simProgress').max = st.T || sim.T || 1;
  $('simProgress').value = st.frame || 0;
  if (st.status === 'running' || st.status === 'done') {
    if (!sim.meta) await loadMeta(id);
    if (sim.meta && (st.frame || 0) > sim.nFrames) await loadFrames(id);
  }
  if (st.status === 'starting') $('simInfo').textContent = 'Φόρτωση του μοντέλου και σχεδιασμός διαδρομών (D* Lite)…';
  if (st.status === 'running') $('simInfo').textContent = `Τρέχει: ${st.frame} / ${st.T} s · διέφυγαν ${st.escaped ?? 0}`;
  if (st.status === 'done') {
    await loadFrames(id);
    finish(await getJson(fileUrl(id, 'result.json')));
  } else if (st.status === 'failed' || st.status === 'cancelled') {
    clearInterval(sim.timer);
    $('simRun').disabled = false;
    $('simCancel').classList.add('hidden');
    $('simProgress').classList.add('hidden');
    $('simInfo').textContent = st.status === 'cancelled' ? 'Ακυρώθηκε.' : 'Αποτυχία: ' + (st.error || '');
  }
}

function finish(r) {
  clearInterval(sim.timer);
  $('simRun').disabled = false;
  $('simCancel').classList.add('hidden');
  $('simProgress').classList.add('hidden');
  const mean = r.mean_escape_time == null ? '–' : r.mean_escape_time.toFixed(0) + ' s';
  $('simInfo').textContent = `Τέλος: διέφυγαν ${r.escaped} / ${r.n_agents} · ακινητοποιημένοι (TL ≥ 3): ${r.incapacitated} · μέσος χρόνος διαφυγής ${mean} · χρόνος υπολογισμού ${r.total_time} s`
    + (r.no_path ? ` · ${r.no_path} agents χωρίς διαδρομή` : '');
  $('dlCsv').href = fileUrl(sim.id, 'trajectories.csv');
  $('dlResult').href = fileUrl(sim.id, 'result.json');
  $('simDownloads').classList.remove('hidden');
}

// ---------- δεδομένα ----------
async function loadMeta(id) {
  const meta = await getJson(fileUrl(id, 'meta.json'));
  if (id !== sim.id) return;
  sim.meta = meta; sim.n = meta.n_agents; sim.stride = meta.stride; sim.T = meta.T;
  sim.cm = await Promise.all(Array.from({ length: meta.num_cms }, (_, k) => loadCm(fileUrl(id, `cm_${k + 1}.png`))));
  $('simbar').classList.remove('hidden');
  $('simSlider').max = 0;
}

// Το CM ως έγχρωμο overlay: κίτρινο → κόκκινο, διαφάνεια ανάλογη της συγκέντρωσης (0 = διάφανο)
function loadCm(url) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      const x = c.getContext('2d');
      x.drawImage(img, 0, 0);
      const d = x.getImageData(0, 0, c.width, c.height);
      for (let i = 0; i < d.data.length; i += 4) {
        const v = d.data[i] / 255;
        d.data[i] = 255; d.data[i + 1] = Math.round(235 * (1 - v)); d.data[i + 2] = 0;
        d.data[i + 3] = v < 0.02 ? 0 : Math.round(255 * (0.25 + 0.55 * v));
      }
      x.putImageData(d, 0, 0);
      resolve(c);
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

async function loadFrames(id) {
  const buf = await (await fetch(fileUrl(id, 'frames.bin') + '?t=' + Date.now())).arrayBuffer();
  if (id !== sim.id || !sim.meta) return;
  const per = sim.n * sim.stride;
  const nFrames = Math.floor(buf.byteLength / 4 / per);
  if (nFrames <= sim.nFrames) return;
  sim.data = new Float32Array(buf, 0, nFrames * per);
  for (let f = sim.nFrames; f < nFrames; f++) {
    let esc = 0, dead = 0;
    for (let k = 0; k < sim.n; k++) {
      const o = f * per + k * sim.stride;
      if (sim.data[o + 3] > 0.5) esc++; else if (sim.data[o + 2] >= 3) dead++;
    }
    sim.esc[f] = esc; sim.dead[f] = dead;
  }
  sim.nFrames = nFrames;
  $('simSlider').max = nFrames - 1;
  if (sim.follow) setFrame(nFrames - 1);
  else updateBar();
}

// ---------- αναπαραγωγή ----------
export function setFrame(f) {
  sim.frame = Math.max(0, Math.min(f, sim.nFrames - 1));
  $('simSlider').value = sim.frame;
  updateBar();
  redraw();
}

function updateBar() {
  if (!sim.nFrames) return;
  const f = sim.frame;
  $('simTime').textContent = `t = ${f + 1} s · διέφυγαν ${sim.esc[f]} / ${sim.n} · TL ≥ 3: ${sim.dead[f]}`;
  $('simPlay').textContent = sim.playing ? '⏸' : '▶';
  drawCharts();
}

export function togglePlay() {
  if (!sim.nFrames) return;
  sim.playing = !sim.playing;
  sim.follow = false;
  if (sim.playing && sim.frame >= sim.nFrames - 1) sim.frame = 0;
  sim.lastTick = performance.now();
  updateBar();
  if (sim.playing) requestAnimationFrame(tick);
}

function tick(now) {
  if (!sim.playing) return;
  const adv = Math.floor((now - sim.lastTick) / 1000 * FPS * sim.speed);
  if (adv >= 1) {
    sim.lastTick = now;
    const target = sim.frame + adv;
    if (target >= sim.nFrames - 1) {
      setFrame(sim.nFrames - 1);
      if (sim.status === 'running') sim.follow = true;     // ζωντανά: ακολουθεί το τελευταίο διαθέσιμο frame
      sim.playing = false;
      updateBar();
      return;
    }
    setFrame(target);
  }
  requestAnimationFrame(tick);
}

export function userScrub(f) { sim.follow = false; sim.playing = false; setFrame(f); }
export function setSpeed(v) { sim.speed = v; }

// ---------- σχεδίαση στον καμβά της εικόνας ----------
const personColor = (tl, esc) => (esc ? '#7f8c9a' : tl >= 3 ? '#ff3b30' : tl <= 1 ? '#34c759' : '#ff9f0a');

// Σχεδιάζεται στο σύστημα συντεταγμένων της εικόνας· g = grid του project (x0, y0, f, w, h)
export function drawSim(ctx, vp, g, showCm) {
  if (!sim.data || !g) return;
  const px = 1 / vp.scale, per = sim.n * sim.stride, base = sim.frame * per;
  const X = col => g.x0 + col / g.f, Y = row => g.y0 + row / g.f;
  if (showCm && sim.cm.length) {
    const t = sim.frame;                                  // dt = 1 s: το CM αλλάζει στους χρόνους εναλλαγής του meta
    const idx = Math.max(0, sim.meta.cm_switch_times.findLastIndex(s => s <= t));
    const cm = sim.cm[idx];
    if (cm) { ctx.imageSmoothingEnabled = false; ctx.drawImage(cm, g.x0, g.y0, g.w / g.f, g.h / g.f); }
  }
  ctx.lineWidth = 2 * px;
  sim.meta.goals.forEach(e => {                           // έξοδοι: κόκκινοι κύκλοι με την ακτίνα διαφυγής
    ctx.beginPath(); ctx.arc(X(e.col), Y(e.row), sim.meta.goal_radius / g.f, 0, 7);
    ctx.strokeStyle = 'rgba(255,59,48,.9)'; ctx.stroke();
  });
  const r = Math.max(3 * px, 2.2 / g.f);
  for (let k = 0; k < sim.n; k++) {
    const o = base + k * sim.stride;
    ctx.beginPath(); ctx.arc(X(sim.data[o]), Y(sim.data[o + 1]), r, 0, 7);
    ctx.fillStyle = personColor(sim.data[o + 2], sim.data[o + 3] > 0.5); ctx.fill();
    ctx.lineWidth = px; ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.stroke();
  }
}

// ---------- γραφήματα ----------
const HIST_EDGES = 8;                                      // bins: TL = 0, (0,3) ανά 0.5 (6 bins), TL ≥ 3
const HIST_COLORS = ['#7a7a7a', '#3b82f6', '#3b82f6', '#eab308', '#eab308', '#f97316', '#f97316', '#dc2626'];
const HIST_LABELS = ['0', '<.5', '<1', '<1.5', '<2', '<2.5', '<3', '≥3'];

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888'; }

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth || 300, h = canvas.clientHeight || 120;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const c = canvas.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  c.font = '10px system-ui, sans-serif';
  return { c, w, h };
}

export function drawCharts() {
  const ink = css('--muted'), line = css('--line');
  // 1) διαφυγή και ακινητοποίηση σε σχέση με τον χρόνο
  {
    const { c, w, h } = setup($('chartEsc'));
    const L = 28, B = 16, plotW = w - L - 6, plotH = h - B - 6, T = Math.max(sim.T, sim.nFrames, 1), n = Math.max(sim.n, 1);
    c.strokeStyle = line; c.strokeRect(L, 6, plotW, plotH);
    c.fillStyle = ink; c.textAlign = 'right'; c.fillText(String(n), L - 3, 14); c.fillText('0', L - 3, 6 + plotH);
    c.textAlign = 'center'; c.fillText('0', L, h - 3); c.fillText(String(T) + ' s', L + plotW - 8, h - 3);
    [[sim.esc, '#7f8c9a'], [sim.dead, '#dc2626']].forEach(([arr, col]) => {
      if (!arr.length) return;
      c.beginPath();
      arr.forEach((v, f) => { const x = L + f / T * plotW, y = 6 + plotH * (1 - v / n); f ? c.lineTo(x, y) : c.moveTo(x, y); });
      c.strokeStyle = col; c.lineWidth = 2; c.stroke();
    });
    if (sim.nFrames) { const x = L + sim.frame / T * plotW; c.strokeStyle = ink; c.lineWidth = 1; c.beginPath(); c.moveTo(x, 6); c.lineTo(x, 6 + plotH); c.stroke(); }
    c.textAlign = 'left'; c.fillStyle = '#7f8c9a'; c.fillText('διέφυγαν', L + 4, 18); c.fillStyle = '#dc2626'; c.fillText('TL ≥ 3', L + 58, 18);
  }
  // 2) κατανομή τοξικού φορτίου στο τρέχον frame (bins όπως στο ιστόγραμμα του σεναρίου)
  {
    const { c, w, h } = setup($('chartTL'));
    const counts = new Array(HIST_EDGES).fill(0);
    if (sim.data) {
      const base = sim.frame * sim.n * sim.stride;
      for (let k = 0; k < sim.n; k++) {
        const tl = sim.data[base + k * sim.stride + 2];
        counts[tl <= 0 ? 0 : tl >= 3 ? 7 : 1 + Math.min(5, Math.floor(tl / 0.5))]++;
      }
    }
    const L = 28, B = 16, plotW = w - L - 6, plotH = h - B - 6, n = Math.max(sim.n, 1), bw = plotW / HIST_EDGES;
    c.strokeStyle = line; c.strokeRect(L, 6, plotW, plotH);
    c.fillStyle = ink; c.textAlign = 'right'; c.fillText(String(n), L - 3, 14); c.fillText('0', L - 3, 6 + plotH);
    counts.forEach((v, i) => {
      c.fillStyle = HIST_COLORS[i]; const bh = plotH * v / n;
      c.fillRect(L + i * bw + 3, 6 + plotH - bh, bw - 6, bh);
      c.fillStyle = ink; c.textAlign = 'center'; c.fillText(HIST_LABELS[i], L + (i + 0.5) * bw, h - 3);
      if (v) c.fillText(String(v), L + (i + 0.5) * bw, 6 + plotH - bh - 2);
    });
  }
}
