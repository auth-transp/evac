// Σχεδίαση του καμβά: εικόνα, χάρτης κλάσεων / heightmap, περιοχή, έξοδοι, ζώνες spawn, προσομοίωση, προσχέδιο
import { drawSim } from './sim.js';

// Γεωμετρία ορθογωνίου εξόδου (σε συντεταγμένες εικόνας) από τμήμα p1–p2 και πάχος σε pixels της εικόνας
export function exitCorners(p1, p2, thicknessPx) {
  const dx = p2[0] - p1[0], dy = p2[1] - p1[1], len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len * thicknessPx / 2, ny = dx / len * thicknessPx / 2;
  return [[p1[0] + nx, p1[1] + ny], [p2[0] + nx, p2[1] + ny], [p2[0] - nx, p2[1] - ny], [p1[0] - nx, p1[1] - ny]];
}

function path(ctx, pts, close) {
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  if (close) ctx.closePath();
}

export function render(canvas, vp, s) {
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!s.img) return;
  const d = vp.dpr, px = 1 / vp.scale;                        // px = 1 css pixel σε συντεταγμένες εικόνας
  ctx.setTransform(vp.scale * d, 0, 0, vp.scale * d, vp.tx * d, vp.ty * d);
  ctx.imageSmoothingEnabled = vp.scale < 1;
  ctx.drawImage(s.img, 0, 0);

  if (s.grid) {
    const g = s.grid;
    ctx.imageSmoothingEnabled = false;
    if (s.view === 'labels' && s.labelsCanvas) ctx.drawImage(s.labelsCanvas, g.x0, g.y0, g.w / g.f, g.h / g.f);
    if (s.view === 'heightmap' && s.hmImg) ctx.drawImage(s.hmImg, g.x0, g.y0, g.w / g.f, g.h / g.f);
  }

  // περιοχή εγκατάστασης
  ctx.lineWidth = 2 * px;
  s.region.forEach(r => {
    path(ctx, r.points, true);
    ctx.strokeStyle = r.mode === 'add' ? '#2ecc71' : '#ff5a5f';
    ctx.setLineDash(r.mode === 'add' ? [] : [6 * px, 4 * px]);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // ζώνες spawn
  s.spawn.forEach(z => {
    path(ctx, z.points, true);
    ctx.fillStyle = 'rgba(56,132,255,.28)'; ctx.fill();
    ctx.strokeStyle = '#3884ff'; ctx.stroke();
  });

  // έξοδοι
  s.exits.forEach(e => {
    path(ctx, exitCorners(e.p1, e.p2, e.thickness_m / (s.srcMpp || 1)), true);
    ctx.fillStyle = 'rgba(255,214,10,.55)'; ctx.fill();
    ctx.strokeStyle = '#ffd60a'; ctx.stroke();
  });

  // agents, concentration map και έξοδοι της προσομοίωσης
  drawSim(ctx, vp, s.grid, s.showCm);

  // γραμμή αναφοράς κλίμακας
  if (s.refLine) {
    path(ctx, s.refLine, false);
    ctx.strokeStyle = '#ff7a00'; ctx.lineWidth = 3 * px; ctx.stroke();
    s.refLine.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], 4 * px, 0, 7); ctx.fillStyle = '#ff7a00'; ctx.fill(); });
  }

  // προσχέδιο (πολύγωνο ή τμήμα εξόδου που σχεδιάζεται)
  const dr = s.draft;
  if (dr) {
    const pts = dr.cursor ? [...dr.points, dr.cursor] : dr.points;
    ctx.lineWidth = 2 * px;
    ctx.strokeStyle = dr.kind === 'region' ? (dr.mode === 'add' ? '#2ecc71' : '#ff5a5f') : '#ffffff';
    ctx.setLineDash([5 * px, 4 * px]);
    path(ctx, pts, false); ctx.stroke();
    ctx.setLineDash([]);
    dr.points.forEach((p, i) => { ctx.beginPath(); ctx.arc(p[0], p[1], (i ? 3 : 5) * px, 0, 7); ctx.fillStyle = '#fff'; ctx.fill(); });
  }

  // δείκτης πινέλου
  if (s.brushCursor && s.tool === 'brush' && s.grid) {
    ctx.beginPath(); ctx.arc(s.brushCursor[0], s.brushCursor[1], Math.max(s.brushSize / 2 / s.grid.f, px), 0, 7);   // μέγεθος = διάμετρος σε cells
    ctx.strokeStyle = '#fff'; ctx.lineWidth = px; ctx.stroke();
  }
}
