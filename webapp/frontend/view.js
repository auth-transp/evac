// Viewport: zoom/pan του καμβά και μετατροπή συντεταγμένων οθόνης <-> εικόνας

export class Viewport {
  constructor(canvas, onChange) {
    this.canvas = canvas;
    this.onChange = onChange;           // καλείται όταν χρειάζεται επανασχεδίαση
    this.scale = 1; this.tx = 0; this.ty = 0;
    this.tool = null;                   // {down, move, up, click, dblclick}(p, e) με p = συντεταγμένες εικόνας
    this.spaceDown = false;
    this.pan = null;
    this.press = null;
    new ResizeObserver(() => { this.resize(); onChange(); }).observe(canvas.parentElement);
    canvas.addEventListener('wheel', e => this.wheel(e), { passive: false });
    canvas.addEventListener('pointerdown', e => this.pointerDown(e));
    canvas.addEventListener('pointermove', e => this.pointerMove(e));
    canvas.addEventListener('pointerup', e => this.pointerUp(e));
    canvas.addEventListener('dblclick', e => this.tool?.dblclick?.(this.toImage(e), e));
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    window.addEventListener('keydown', e => { if (e.code === 'Space' && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { this.spaceDown = true; e.preventDefault(); } });
    window.addEventListener('keyup', e => { if (e.code === 'Space') this.spaceDown = false; });
    this.resize();
  }

  resize() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(r.width * this.dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * this.dpr));
  }

  // συντεταγμένες οθόνης (css pixels) από event
  screen(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }
  toImage(e) { const [sx, sy] = this.screen(e); return [(sx - this.tx) / this.scale, (sy - this.ty) / this.scale]; }
  toScreen(p) { return [p[0] * this.scale + this.tx, p[1] * this.scale + this.ty]; }

  fit(w, h) {
    const r = this.canvas.parentElement.getBoundingClientRect();
    this.scale = Math.min(r.width / w, r.height / h) * 0.95;
    this.tx = (r.width - w * this.scale) / 2;
    this.ty = (r.height - h * this.scale) / 2;
    this.onChange();
  }

  wheel(e) {
    e.preventDefault();
    const [sx, sy] = this.screen(e);
    const k = Math.exp(-e.deltaY * 0.0015);
    const ns = Math.min(Math.max(this.scale * k, 0.02), 60);
    this.tx = sx - (sx - this.tx) * ns / this.scale;
    this.ty = sy - (sy - this.ty) * ns / this.scale;
    this.scale = ns;
    this.onChange();
  }

  pointerDown(e) {
    this.canvas.setPointerCapture(e.pointerId);
    if (e.button === 1 || e.button === 2 || this.spaceDown || !this.tool) {
      this.pan = { sx: e.clientX, sy: e.clientY, tx: this.tx, ty: this.ty };
      return;
    }
    if (e.button !== 0) return;
    this.press = { x: e.clientX, y: e.clientY, moved: false };
    this.tool.down?.(this.toImage(e), e);
  }

  pointerMove(e) {
    if (this.pan) {
      this.tx = this.pan.tx + e.clientX - this.pan.sx;
      this.ty = this.pan.ty + e.clientY - this.pan.sy;
      this.onChange();
      return;
    }
    if (this.press && Math.hypot(e.clientX - this.press.x, e.clientY - this.press.y) > 4) this.press.moved = true;
    this.tool?.move?.(this.toImage(e), e, !!this.press);
  }

  pointerUp(e) {
    if (this.pan) { this.pan = null; return; }
    if (!this.press) return;
    const moved = this.press.moved;
    this.press = null;
    this.tool?.up?.(this.toImage(e), e);
    if (!moved) this.tool?.click?.(this.toImage(e), e);
  }
}
