// ============================================================
// Neo-Noir Auto Theme — 1440 Minute Resolution
// Time-dynamic hourly palette × per-minute background gradients
// ============================================================

// --- Interpolation Helpers (module-private) ---
const _lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const _clamp = (v: number) => Math.min(Math.max(Math.round(v), 0), 255);
const _lerpRGB = (c1: number[], c2: number[], t: number) => c1.map((v, i) => _clamp(_lerp(v, c2[i], t)));
const _hex = (rgb: number[]) =>
  "#" + rgb.map((v) => v.toString(16).padStart(2, "0")).join("");
const _r2 = (n: number) => Math.round(n * 100) / 100;
const _ts = (m: number) => `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`;
const _fill = (style: string) =>
  Array.from({ length: 1440 }, (_, i) => ({ time: _ts(i), style }));

type Keyframe = {
  r1: number[];
  a1: number;
  r2: number[];
  a2: number;
  g1: number[];
  g2: number[];
  g3: number[];
  s1: number;
  s2: number;
  mp: number;
};

// --- 24 Hourly Neo-Noir Background Keyframes ---
// r1/a1 : primary radial  (circle at top)         — colour + opacity
// r2/a2 : secondary radial (circle at 20% 20%)    — colour + opacity
// g1/g2/g3 : linear-gradient stops (top / mid / bottom)
// s1/s2 : radial spread %
// mp    : linear-gradient mid-point %
const _K: Keyframe[] = [
  /* 00 — Midnight Abyss          */ {
    r1: [30, 27, 75],
    a1: 0.2,
    r2: [34, 211, 238],
    a2: 0.06,
    g1: [4, 6, 20],
    g2: [3, 5, 14],
    g3: [0, 0, 2],
    s1: 32,
    s2: 24,
    mp: 42,
  },
  /* 01 — Deep Void               */ {
    r1: [30, 27, 75],
    a1: 0.16,
    r2: [34, 211, 238],
    a2: 0.05,
    g1: [3, 5, 18],
    g2: [2, 4, 12],
    g3: [0, 0, 2],
    s1: 30,
    s2: 24,
    mp: 42,
  },
  /* 02 — Abyssal Quiet           */ {
    r1: [24, 24, 68],
    a1: 0.14,
    r2: [20, 184, 166],
    a2: 0.05,
    g1: [3, 4, 16],
    g2: [2, 3, 10],
    g3: [0, 0, 2],
    s1: 30,
    s2: 24,
    mp: 42,
  },
  /* 03 — System Stir (emerald)   */ {
    r1: [24, 24, 68],
    a1: 0.14,
    r2: [52, 211, 153],
    a2: 0.06,
    g1: [2, 4, 14],
    g2: [2, 3, 10],
    g3: [0, 0, 2],
    s1: 30,
    s2: 25,
    mp: 42,
  },
  /* 04 — Purple Dawn Bleed       */ {
    r1: [88, 28, 135],
    a1: 0.16,
    r2: [52, 211, 153],
    a2: 0.08,
    g1: [6, 4, 18],
    g2: [4, 3, 14],
    g3: [1, 0, 4],
    s1: 32,
    s2: 25,
    mp: 42,
  },
  /* 05 — Fuchsia Awakening       */ {
    r1: [192, 38, 211],
    a1: 0.18,
    r2: [34, 211, 238],
    a2: 0.1,
    g1: [10, 6, 24],
    g2: [8, 4, 18],
    g3: [2, 0, 6],
    s1: 34,
    s2: 25,
    mp: 42,
  },
  /* 06 — Warm Refuge Dawn        */ {
    r1: [245, 158, 11],
    a1: 0.18,
    r2: [34, 211, 238],
    a2: 0.12,
    g1: [14, 10, 26],
    g2: [10, 8, 22],
    g3: [2, 1, 6],
    s1: 35,
    s2: 25,
    mp: 42,
  },
  /* 07 — Orange Neon Rise        */ {
    r1: [251, 146, 60],
    a1: 0.22,
    r2: [34, 211, 238],
    a2: 0.14,
    g1: [16, 12, 28],
    g2: [12, 10, 24],
    g3: [3, 2, 8],
    s1: 36,
    s2: 26,
    mp: 44,
  },
  /* 08 — Yellow Wash + Fuchsia   */ {
    r1: [253, 224, 71],
    a1: 0.2,
    r2: [217, 70, 239],
    a2: 0.1,
    g1: [14, 14, 30],
    g2: [10, 10, 24],
    g3: [2, 2, 8],
    s1: 36,
    s2: 26,
    mp: 44,
  },
  /* 09 — Cyan Atmospheric        */ {
    r1: [34, 211, 238],
    a1: 0.22,
    r2: [52, 211, 153],
    a2: 0.12,
    g1: [8, 16, 36],
    g2: [6, 12, 28],
    g3: [1, 3, 10],
    s1: 38,
    s2: 28,
    mp: 46,
  },
  /* 10 — Bright Cyan Flood       */ {
    r1: [56, 189, 248],
    a1: 0.26,
    r2: [34, 211, 238],
    a2: 0.16,
    g1: [6, 20, 42],
    g2: [4, 14, 32],
    g3: [1, 4, 14],
    s1: 38,
    s2: 28,
    mp: 48,
  },
  /* 11 — Peak Cyan + Teal        */ {
    r1: [14, 165, 233],
    a1: 0.28,
    r2: [20, 184, 166],
    a2: 0.18,
    g1: [4, 22, 48],
    g2: [3, 16, 36],
    g3: [1, 4, 16],
    s1: 40,
    s2: 30,
    mp: 50,
  },
  /* 12 — Midday Surveillance     */ {
    r1: [14, 165, 233],
    a1: 0.3,
    r2: [56, 189, 248],
    a2: 0.18,
    g1: [4, 20, 46],
    g2: [3, 15, 34],
    g3: [1, 4, 14],
    s1: 40,
    s2: 30,
    mp: 55,
  },
  /* 13 — Afternoon Shift         */ {
    r1: [14, 165, 233],
    a1: 0.26,
    r2: [99, 102, 241],
    a2: 0.14,
    g1: [5, 18, 42],
    g2: [4, 14, 32],
    g3: [1, 3, 12],
    s1: 38,
    s2: 28,
    mp: 50,
  },
  /* 14 — Indigo Creeping          */ {
    r1: [99, 102, 241],
    a1: 0.24,
    r2: [139, 92, 246],
    a2: 0.14,
    g1: [6, 14, 38],
    g2: [5, 12, 28],
    g3: [1, 2, 10],
    s1: 36,
    s2: 26,
    mp: 48,
  },
  /* 15 — Purple Atmosphere       */ {
    r1: [139, 92, 246],
    a1: 0.24,
    r2: [217, 70, 239],
    a2: 0.14,
    g1: [10, 10, 36],
    g2: [8, 8, 28],
    g3: [2, 1, 10],
    s1: 36,
    s2: 26,
    mp: 45,
  },
  /* 16 — Fuchsia Neon Rise       */ {
    r1: [217, 70, 239],
    a1: 0.24,
    r2: [244, 114, 182],
    a2: 0.16,
    g1: [14, 8, 38],
    g2: [12, 6, 30],
    g3: [3, 1, 10],
    s1: 35,
    s2: 25,
    mp: 44,
  },
  /* 17 — Magenta + Amber         */ {
    r1: [217, 70, 239],
    a1: 0.26,
    r2: [249, 115, 22],
    a2: 0.18,
    g1: [18, 6, 40],
    g2: [16, 4, 32],
    g3: [4, 1, 10],
    s1: 35,
    s2: 25,
    mp: 42,
  },
  /* 18 — Sunset Off-World Neon   */ {
    r1: [249, 115, 22],
    a1: 0.24,
    r2: [217, 70, 239],
    a2: 0.18,
    g1: [22, 6, 42],
    g2: [18, 4, 34],
    g3: [4, 1, 12],
    s1: 35,
    s2: 25,
    mp: 42,
  },
  /* 19 — Rose + Purple Descent   */ {
    r1: [225, 29, 72],
    a1: 0.22,
    r2: [168, 85, 247],
    a2: 0.16,
    g1: [20, 4, 38],
    g2: [16, 3, 30],
    g3: [4, 1, 10],
    s1: 34,
    s2: 25,
    mp: 42,
  },
  /* 20 — Purple Night Fall       */ {
    r1: [147, 51, 234],
    a1: 0.22,
    r2: [79, 70, 229],
    a2: 0.14,
    g1: [16, 6, 34],
    g2: [12, 5, 26],
    g3: [3, 1, 8],
    s1: 34,
    s2: 25,
    mp: 42,
  },
  /* 21 — Indigo Night Neon       */ {
    r1: [79, 70, 229],
    a1: 0.22,
    r2: [34, 211, 238],
    a2: 0.1,
    g1: [12, 8, 30],
    g2: [8, 6, 22],
    g3: [2, 1, 6],
    s1: 33,
    s2: 25,
    mp: 42,
  },
  /* 22 — Settling Darkness       */ {
    r1: [59, 130, 246],
    a1: 0.2,
    r2: [34, 211, 238],
    a2: 0.08,
    g1: [8, 8, 26],
    g2: [6, 6, 18],
    g3: [1, 1, 4],
    s1: 32,
    s2: 24,
    mp: 42,
  },
  /* 23 — Pre-Midnight Return     */ {
    r1: [37, 99, 235],
    a1: 0.22,
    r2: [34, 211, 238],
    a2: 0.06,
    g1: [6, 8, 24],
    g2: [4, 6, 16],
    g3: [1, 1, 4],
    s1: 32,
    s2: 24,
    mp: 42,
  },
];

function _lerpKF(k1: Keyframe, k2: Keyframe, t: number): Keyframe {
  return {
    r1: _lerpRGB(k1.r1, k2.r1, t),
    a1: _r2(_lerp(k1.a1, k2.a1, t)),
    r2: _lerpRGB(k1.r2, k2.r2, t),
    a2: _r2(_lerp(k1.a2, k2.a2, t)),
    g1: _lerpRGB(k1.g1, k2.g1, t),
    g2: _lerpRGB(k1.g2, k2.g2, t),
    g3: _lerpRGB(k1.g3, k2.g3, t),
    s1: Math.round(_lerp(k1.s1, k2.s1, t)),
    s2: Math.round(_lerp(k1.s2, k2.s2, t)),
    mp: Math.round(_lerp(k1.mp, k2.mp, t)),
  };
}

function _kfToStyle(kf: Keyframe) {
  return {
    backgroundImage: `radial-gradient(circle at top, rgba(${kf.r1.join(",")}, ${kf.a1}), transparent ${kf.s1}%), radial-gradient(circle at 20% 20%, rgba(${kf.r2.join(",")}, ${kf.a2}), transparent ${kf.s2}%), linear-gradient(180deg, ${_hex(kf.g1)} 0%, ${_hex(kf.g2)} ${kf.mp}%, ${_hex(kf.g3)} 100%)`,
  };
}

// --- Generate 1440 minute entries via keyframe interpolation ---
const _bg: { time: string; style: { backgroundImage: string } }[] = [];
for (let m = 0; m < 1440; m++) {
  const hour = m / 60;
  const hi = Math.floor(hour);
  const t = hour - hi;
  const ni = (hi + 1) % 24;
  const kf = t === 0 ? _K[hi] : _lerpKF(_K[hi], _K[ni], t);
  _bg.push({ time: _ts(m), style: _kfToStyle(kf) });
}

export const backgroundColorAutoStyle = _bg;

// ============================================================
// HOURLY ACCENT PALETTE SYSTEM
// ============================================================
// Every hour, accent colors rotate through curated Neo-Noir palettes.
// Background gradients change per-minute (interpolated above).
// Accent tokens change per-hour (Tailwind class swaps).
//
// 4 palette roles per hour:
//   pri  — Primary neon accent (CTAs, borders, neon text, glows)
//   sec  — Secondary accent (contrasting neon, rings)
//   warm — Warm refuge tones (amber/orange/rose family)
//   hud  — HUD/terminal tones (teal/emerald/cyan family)

// --- Pre-computed Color Palettes ---
// All class names are literal strings so Tailwind's JIT scanner detects them.
// Properties: ov(overlay) ovl(overlay-light) sol(solid) cta(cta-button)
//   neo(neon-overlay) hov(hud-overlay) hovl(hud-overlay-light)
//   hCta(hover-cta) h20(hover-accent/20) hBr(hover-bright) hTx(hover-text)
//   tx(text) txB(text-bright) txD(text-deep)
//   b40(border/40) r20(ring/20) d20(divide/20) b30(border/30) bS(border-solid) rS(ring-solid)
//   glo(glow) gloS(glow-strong) ctaG(cta-glow) wOv(warm-overlay-compound)

export type PaletteTokens = {
  ov: string;
  ovl: string;
  sol: string;
  cta: string;
  neo: string;
  hov: string;
  hovl: string;
  hCta: string;
  h20: string;
  hBr: string;
  hTx: string;
  tx: string;
  txB: string;
  txD: string;
  b40: string;
  r20: string;
  d20: string;
  b30: string;
  bS: string;
  rS: string;
  glo: string;
  gloS: string;
  ctaG: string;
  wOv: string;
};

const _C: Record<string, PaletteTokens> = {
  cyan: {
    ov: "bg-cyan-900/20",
    ovl: "bg-cyan-900/10",
    sol: "bg-cyan-400",
    cta: "bg-cyan-500",
    neo: "bg-cyan-500/10",
    hov: "bg-cyan-950/40",
    hovl: "bg-cyan-900/20",
    hCta: "hover:bg-cyan-600",
    h20: "hover:bg-cyan-400/20",
    hBr: "hover:bg-cyan-300",
    hTx: "hover:text-cyan-300",
    tx: "text-cyan-400",
    txB: "text-cyan-300",
    txD: "text-cyan-500",
    b40: "border-cyan-500/40",
    r20: "ring-cyan-400/20",
    d20: "divide-cyan-500/20",
    b30: "border-cyan-400/30",
    bS: "border-cyan-500",
    rS: "ring-1 ring-cyan-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(34,211,238,0.9)] shadow-[0_0_15px_rgba(34,211,238,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(34,211,238,1)] shadow-[0_0_20px_rgba(34,211,238,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(34,211,238,0.3)]",
    wOv: "bg-cyan-500/20 border border-cyan-500/50 shadow-lg shadow-cyan-500/20",
  },
  indigo: {
    ov: "bg-indigo-900/20",
    ovl: "bg-indigo-900/10",
    sol: "bg-indigo-400",
    cta: "bg-indigo-500",
    neo: "bg-indigo-500/10",
    hov: "bg-indigo-950/40",
    hovl: "bg-indigo-900/20",
    hCta: "hover:bg-indigo-600",
    h20: "hover:bg-indigo-400/20",
    hBr: "hover:bg-indigo-300",
    hTx: "hover:text-indigo-300",
    tx: "text-indigo-400",
    txB: "text-indigo-300",
    txD: "text-indigo-500",
    b40: "border-indigo-500/40",
    r20: "ring-indigo-400/20",
    d20: "divide-indigo-500/20",
    b30: "border-indigo-400/30",
    bS: "border-indigo-500",
    rS: "ring-1 ring-indigo-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(129,140,248,0.9)] shadow-[0_0_15px_rgba(129,140,248,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(129,140,248,1)] shadow-[0_0_20px_rgba(129,140,248,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(129,140,248,0.3)]",
    wOv: "bg-indigo-500/20 border border-indigo-500/50 shadow-lg shadow-indigo-500/20",
  },
  fuchsia: {
    ov: "bg-fuchsia-900/20",
    ovl: "bg-fuchsia-900/10",
    sol: "bg-fuchsia-400",
    cta: "bg-fuchsia-500",
    neo: "bg-fuchsia-500/10",
    hov: "bg-fuchsia-950/40",
    hovl: "bg-fuchsia-900/20",
    hCta: "hover:bg-fuchsia-600",
    h20: "hover:bg-fuchsia-400/20",
    hBr: "hover:bg-fuchsia-300",
    hTx: "hover:text-fuchsia-300",
    tx: "text-fuchsia-400",
    txB: "text-fuchsia-300",
    txD: "text-fuchsia-500",
    b40: "border-fuchsia-500/40",
    r20: "ring-fuchsia-400/20",
    d20: "divide-fuchsia-500/20",
    b30: "border-fuchsia-400/30",
    bS: "border-fuchsia-500",
    rS: "ring-1 ring-fuchsia-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(232,121,249,0.9)] shadow-[0_0_15px_rgba(232,121,249,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(232,121,249,1)] shadow-[0_0_20px_rgba(232,121,249,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(232,121,249,0.3)]",
    wOv: "bg-fuchsia-500/20 border border-fuchsia-500/50 shadow-lg shadow-fuchsia-500/20",
  },
  amber: {
    ov: "bg-amber-900/20",
    ovl: "bg-amber-900/10",
    sol: "bg-amber-400",
    cta: "bg-amber-500",
    neo: "bg-amber-500/10",
    hov: "bg-amber-950/40",
    hovl: "bg-amber-900/20",
    hCta: "hover:bg-amber-600",
    h20: "hover:bg-amber-400/20",
    hBr: "hover:bg-amber-300",
    hTx: "hover:text-amber-300",
    tx: "text-amber-400",
    txB: "text-amber-300",
    txD: "text-amber-500",
    b40: "border-amber-500/40",
    r20: "ring-amber-400/20",
    d20: "divide-amber-500/20",
    b30: "border-amber-400/30",
    bS: "border-amber-500",
    rS: "ring-1 ring-amber-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(251,191,36,0.9)] shadow-[0_0_15px_rgba(251,191,36,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(251,191,36,1)] shadow-[0_0_20px_rgba(251,191,36,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(251,191,36,0.3)]",
    wOv: "bg-amber-500/20 border border-amber-500/50 shadow-lg shadow-amber-500/20",
  },
  emerald: {
    ov: "bg-emerald-900/20",
    ovl: "bg-emerald-900/10",
    sol: "bg-emerald-400",
    cta: "bg-emerald-500",
    neo: "bg-emerald-500/10",
    hov: "bg-emerald-950/40",
    hovl: "bg-emerald-900/20",
    hCta: "hover:bg-emerald-600",
    h20: "hover:bg-emerald-400/20",
    hBr: "hover:bg-emerald-300",
    hTx: "hover:text-emerald-300",
    tx: "text-emerald-400",
    txB: "text-emerald-300",
    txD: "text-emerald-500",
    b40: "border-emerald-500/40",
    r20: "ring-emerald-400/20",
    d20: "divide-emerald-500/20",
    b30: "border-emerald-400/30",
    bS: "border-emerald-500",
    rS: "ring-1 ring-emerald-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(52,211,153,0.9)] shadow-[0_0_15px_rgba(52,211,153,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(52,211,153,1)] shadow-[0_0_20px_rgba(52,211,153,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(52,211,153,0.3)]",
    wOv: "bg-emerald-500/20 border border-emerald-500/50 shadow-lg shadow-emerald-500/20",
  },
  purple: {
    ov: "bg-purple-900/20",
    ovl: "bg-purple-900/10",
    sol: "bg-purple-400",
    cta: "bg-purple-500",
    neo: "bg-purple-500/10",
    hov: "bg-purple-950/40",
    hovl: "bg-purple-900/20",
    hCta: "hover:bg-purple-600",
    h20: "hover:bg-purple-400/20",
    hBr: "hover:bg-purple-300",
    hTx: "hover:text-purple-300",
    tx: "text-purple-400",
    txB: "text-purple-300",
    txD: "text-purple-500",
    b40: "border-purple-500/40",
    r20: "ring-purple-400/20",
    d20: "divide-purple-500/20",
    b30: "border-purple-400/30",
    bS: "border-purple-500",
    rS: "ring-1 ring-purple-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(192,132,252,0.9)] shadow-[0_0_15px_rgba(192,132,252,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(192,132,252,1)] shadow-[0_0_20px_rgba(192,132,252,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(192,132,252,0.3)]",
    wOv: "bg-purple-500/20 border border-purple-500/50 shadow-lg shadow-purple-500/20",
  },
  rose: {
    ov: "bg-rose-900/20",
    ovl: "bg-rose-900/10",
    sol: "bg-rose-400",
    cta: "bg-rose-500",
    neo: "bg-rose-500/10",
    hov: "bg-rose-950/40",
    hovl: "bg-rose-900/20",
    hCta: "hover:bg-rose-600",
    h20: "hover:bg-rose-400/20",
    hBr: "hover:bg-rose-300",
    hTx: "hover:text-rose-300",
    tx: "text-rose-400",
    txB: "text-rose-300",
    txD: "text-rose-500",
    b40: "border-rose-500/40",
    r20: "ring-rose-400/20",
    d20: "divide-rose-500/20",
    b30: "border-rose-400/30",
    bS: "border-rose-500",
    rS: "ring-1 ring-rose-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(251,113,133,0.9)] shadow-[0_0_15px_rgba(251,113,133,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(251,113,133,1)] shadow-[0_0_20px_rgba(251,113,133,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(251,113,133,0.3)]",
    wOv: "bg-rose-500/20 border border-rose-500/50 shadow-lg shadow-rose-500/20",
  },
  orange: {
    ov: "bg-orange-900/20",
    ovl: "bg-orange-900/10",
    sol: "bg-orange-400",
    cta: "bg-orange-500",
    neo: "bg-orange-500/10",
    hov: "bg-orange-950/40",
    hovl: "bg-orange-900/20",
    hCta: "hover:bg-orange-600",
    h20: "hover:bg-orange-400/20",
    hBr: "hover:bg-orange-300",
    hTx: "hover:text-orange-300",
    tx: "text-orange-400",
    txB: "text-orange-300",
    txD: "text-orange-500",
    b40: "border-orange-500/40",
    r20: "ring-orange-400/20",
    d20: "divide-orange-500/20",
    b30: "border-orange-400/30",
    bS: "border-orange-500",
    rS: "ring-1 ring-orange-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(251,146,60,0.9)] shadow-[0_0_15px_rgba(251,146,60,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(251,146,60,1)] shadow-[0_0_20px_rgba(251,146,60,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(251,146,60,0.3)]",
    wOv: "bg-orange-500/20 border border-orange-500/50 shadow-lg shadow-orange-500/20",
  },
  sky: {
    ov: "bg-sky-900/20",
    ovl: "bg-sky-900/10",
    sol: "bg-sky-400",
    cta: "bg-sky-500",
    neo: "bg-sky-500/10",
    hov: "bg-sky-950/40",
    hovl: "bg-sky-900/20",
    hCta: "hover:bg-sky-600",
    h20: "hover:bg-sky-400/20",
    hBr: "hover:bg-sky-300",
    hTx: "hover:text-sky-300",
    tx: "text-sky-400",
    txB: "text-sky-300",
    txD: "text-sky-500",
    b40: "border-sky-500/40",
    r20: "ring-sky-400/20",
    d20: "divide-sky-500/20",
    b30: "border-sky-400/30",
    bS: "border-sky-500",
    rS: "ring-1 ring-sky-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(56,189,248,0.9)] shadow-[0_0_15px_rgba(56,189,248,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(56,189,248,1)] shadow-[0_0_20px_rgba(56,189,248,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(56,189,248,0.3)]",
    wOv: "bg-sky-500/20 border border-sky-500/50 shadow-lg shadow-sky-500/20",
  },
  blue: {
    ov: "bg-blue-900/20",
    ovl: "bg-blue-900/10",
    sol: "bg-blue-400",
    cta: "bg-blue-500",
    neo: "bg-blue-500/10",
    hov: "bg-blue-950/40",
    hovl: "bg-blue-900/20",
    hCta: "hover:bg-blue-600",
    h20: "hover:bg-blue-400/20",
    hBr: "hover:bg-blue-300",
    hTx: "hover:text-blue-300",
    tx: "text-blue-400",
    txB: "text-blue-300",
    txD: "text-blue-500",
    b40: "border-blue-500/40",
    r20: "ring-blue-400/20",
    d20: "divide-blue-500/20",
    b30: "border-blue-400/30",
    bS: "border-blue-500",
    rS: "ring-1 ring-blue-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(96,165,250,0.9)] shadow-[0_0_15px_rgba(96,165,250,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(96,165,250,1)] shadow-[0_0_20px_rgba(96,165,250,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(96,165,250,0.3)]",
    wOv: "bg-blue-500/20 border border-blue-500/50 shadow-lg shadow-blue-500/20",
  },
  teal: {
    ov: "bg-teal-900/20",
    ovl: "bg-teal-900/10",
    sol: "bg-teal-400",
    cta: "bg-teal-500",
    neo: "bg-teal-500/10",
    hov: "bg-teal-950/40",
    hovl: "bg-teal-900/20",
    hCta: "hover:bg-teal-600",
    h20: "hover:bg-teal-400/20",
    hBr: "hover:bg-teal-300",
    hTx: "hover:text-teal-300",
    tx: "text-teal-400",
    txB: "text-teal-300",
    txD: "text-teal-500",
    b40: "border-teal-500/40",
    r20: "ring-teal-400/20",
    d20: "divide-teal-500/20",
    b30: "border-teal-400/30",
    bS: "border-teal-500",
    rS: "ring-1 ring-teal-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(45,212,191,0.9)] shadow-[0_0_15px_rgba(45,212,191,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(45,212,191,1)] shadow-[0_0_20px_rgba(45,212,191,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(45,212,191,0.3)]",
    wOv: "bg-teal-500/20 border border-teal-500/50 shadow-lg shadow-teal-500/20",
  },
  violet: {
    ov: "bg-violet-900/20",
    ovl: "bg-violet-900/10",
    sol: "bg-violet-400",
    cta: "bg-violet-500",
    neo: "bg-violet-500/10",
    hov: "bg-violet-950/40",
    hovl: "bg-violet-900/20",
    hCta: "hover:bg-violet-600",
    h20: "hover:bg-violet-400/20",
    hBr: "hover:bg-violet-300",
    hTx: "hover:text-violet-300",
    tx: "text-violet-400",
    txB: "text-violet-300",
    txD: "text-violet-500",
    b40: "border-violet-500/40",
    r20: "ring-violet-400/20",
    d20: "divide-violet-500/20",
    b30: "border-violet-400/30",
    bS: "border-violet-500",
    rS: "ring-1 ring-violet-400/50",
    glo: "drop-shadow-[0_0_8px_rgba(167,139,250,0.9)] shadow-[0_0_15px_rgba(167,139,250,0.4)]",
    gloS: "drop-shadow-[0_0_12px_rgba(167,139,250,1)] shadow-[0_0_20px_rgba(167,139,250,0.3)]",
    ctaG: "shadow-[0_0_15px_rgba(167,139,250,0.3)]",
    wOv: "bg-violet-500/20 border border-violet-500/50 shadow-lg shadow-violet-500/20",
  },
};

// --- Hourly Palette Assignments: [primary, secondary, warm, hud] ---
// Primary accent rotates dramatically. Secondary complements it.
// Warm stays in amber/orange/rose. HUD stays in teal/emerald/cyan.
const _H: [PaletteTokens, PaletteTokens, PaletteTokens, PaletteTokens][] = [
  /* 00 — Midnight Abyss     */ [_C.indigo, _C.violet, _C.amber, _C.teal],
  /* 01 — Deep Void          */ [_C.indigo, _C.violet, _C.amber, _C.teal],
  /* 02 — Abyssal Quiet      */ [_C.teal, _C.cyan, _C.amber, _C.emerald],
  /* 03 — System Stir        */ [_C.emerald, _C.teal, _C.amber, _C.cyan],
  /* 04 — Purple Dawn Bleed  */ [_C.fuchsia, _C.purple, _C.orange, _C.emerald],
  /* 05 — Fuchsia Awakening  */ [_C.fuchsia, _C.rose, _C.orange, _C.teal],
  /* 06 — Warm Refuge Dawn   */ [_C.amber, _C.orange, _C.rose, _C.emerald],
  /* 07 — Orange Neon Rise   */ [_C.orange, _C.amber, _C.rose, _C.teal],
  /* 08 — Yellow Wash        */ [_C.amber, _C.fuchsia, _C.orange, _C.emerald],
  /* 09 — Cyan Atmospheric   */ [_C.cyan, _C.emerald, _C.amber, _C.teal],
  /* 10 — Bright Cyan Flood  */ [_C.cyan, _C.sky, _C.amber, _C.emerald],
  /* 11 — Peak Cyan + Teal   */ [_C.sky, _C.teal, _C.amber, _C.emerald],
  /* 12 — Midday Surveillance*/ [_C.sky, _C.cyan, _C.amber, _C.teal],
  /* 13 — Afternoon Shift    */ [_C.blue, _C.indigo, _C.amber, _C.emerald],
  /* 14 — Indigo Creeping    */ [_C.indigo, _C.violet, _C.amber, _C.teal],
  /* 15 — Purple Atmosphere  */ [_C.purple, _C.fuchsia, _C.orange, _C.emerald],
  /* 16 — Fuchsia Neon Rise  */ [_C.fuchsia, _C.rose, _C.orange, _C.teal],
  /* 17 — Magenta + Amber    */ [_C.rose, _C.fuchsia, _C.amber, _C.emerald],
  /* 18 — Sunset Off-World   */ [_C.orange, _C.rose, _C.amber, _C.teal],
  /* 19 — Rose Descent       */ [_C.rose, _C.purple, _C.amber, _C.emerald],
  /* 20 — Purple Night Fall  */ [_C.purple, _C.indigo, _C.orange, _C.teal],
  /* 21 — Indigo Night Neon  */ [_C.indigo, _C.cyan, _C.amber, _C.emerald],
  /* 22 — Settling Darkness  */ [_C.blue, _C.cyan, _C.amber, _C.teal],
  /* 23 — Pre-Midnight       */ [_C.blue, _C.indigo, _C.amber, _C.teal],
];

// Generate 1440 entries from hourly palette lookup
const _hourly = (fn: (pri: PaletteTokens, sec: PaletteTokens, warm: PaletteTokens, hud: PaletteTokens) => string) =>
  Array.from({ length: 1440 }, (_, m) => {
    const [pri, sec, warm, hud] = _H[Math.floor(m / 60)];
    return { time: _ts(m), style: fn(pri, sec, warm, hud) };
  });

// ============================================================
// COMPATIBILITY EXPORTS (All existing names preserved)
// ============================================================

// --- Glassmorphism Surfaces ---
export const bgCyan400_10 = _hourly((p) => p.ov);
export const bgWhite_5 = _fill("bg-white/5");
export const bgSlate900_70 = _fill("bg-black/40");
export const bgWhite_5_HoverWhite_10 = _fill(
  "bg-white/5 transition hover:bg-white/10",
);

// --- Radial Glow (dual neon — static decorative) ---
export const bgRadialGlow = _fill(
  "bg-[radial-gradient(circle_at_30%_20%,rgba(34,211,238,0.3),transparent_25%),radial-gradient(circle_at_70%_80%,rgba(217,70,239,0.28),transparent_30%)]",
);

// --- Card Glass Surface (noir glass — static) ---
export const bgLinearCard = _fill(
  "bg-[linear-gradient(145deg,rgba(2,6,23,0.96),rgba(9,9,40,0.90))]",
);

// --- System Code / Terminal Greens (HUD palette) ---
export const bgEmerald400_15 = _hourly((p, s, w, h) => h.hov);
export const bgEmerald400_10 = _hourly((p, s, w, h) => h.hov);
export const bgemerald_400_10 = _hourly((p, s, w, h) => h.hov);
export const bgemerald_400_15 = _hourly((p, s, w, h) => h.hov);
// --- Abyssal Base Surfaces (static neutral darks) ---
export const bgSlate950_50 = _fill("bg-slate-950/50");
export const bgslate_950_50 = _fill("bg-slate-950/50");
export const bgslate_950_65 = _fill("bg-slate-950");
export const bgslate_950_95 = _fill("bg-zinc-950");
export const bgslate_950_30 = _fill("bg-indigo-950/30");
export const bgslate_950_45 = _fill("bg-slate-950/45");
export const bgslate_950_60 = _fill("bg-slate-950/60");
export const bgslate_900_70 = _fill("bg-black/40");
export const bgslate_900_50 = _fill("bg-slate-950/50");
export const bgslate_700 = _fill("bg-zinc-950");
// --- Off-World Neon Accents (primary palette) ---
export const bgSky400_10 = _hourly((p) => p.ov);
export const bgsky_400_10 = _hourly((p) => p.ov);
export const bgcyan_400 = _hourly((p) => p.sol);
export const bgcyan_400_15 = _hourly((p) => p.ov);
export const bgcyan_400_10 = _hourly((p) => p.ov);
export const bgcyan_400_5 = _hourly((p) => p.ovl);
// --- Warm Refuge (warm palette) ---
export const bgamber_400_10 = _hourly((p, s, w) => w.ov);
// --- Fuchsia / Rose (secondary palette) ---
export const bgrose_400_10 = _hourly((p, s) => s.ovl);
// --- Primary Action Buttons ---
export const bgwhite = _fill("bg-white");
export const bgslate_100 = _fill("bg-zinc-950");
export const bgWhiteHoverSlate200 = _hourly(
  (p) => `${p.sol} transition ${p.hBr} ${p.ctaG}`,
);
// --- Translucent Fog Layers (static neutrals) ---
export const bgwhite_3 = _fill("bg-white/5");
export const bgwhite_4 = _fill("bg-white/5");
export const bgwhite_5 = _fill("bg-white/5");
export const bgwhite_10 = _fill("bg-black/40");
export const bgwhite_20 = _fill("bg-white/5");
// --- CTA Buttons (primary palette) ---
export const bgblue_600 = _hourly((p) => p.cta);
export const bgblue_700_20 = _hourly((p) => p.ov);
export const bggreen_700_10 = _hourly((p, s, w, h) => h.hov);
// --- Cyan Interaction (primary palette) ---
export const bgCyan400_10_HoverCyan400_20 = _hourly(
  (p) => `${p.ov} transition ${p.h20}`,
);
// --- Hover States ---
export const HoverBgslate_200 = _fill("hover:bg-zinc-900");
export const HoverBgwhite_10 = _fill("hover:bg-white/10");
export const HoverBgwhite_20 = _fill("hover:bg-white/10");
export const HoverBgwhite_5 = _fill("hover:bg-white/5");
export const HoverBgblue_700 = _hourly((p) => p.hCta);
export const HoverBgcyan_300 = _hourly((p) => p.hBr);
export const HoverBgcyan_400_20 = _hourly((p) => p.h20);

// ============================================================
// I. THE EXPANDED NEO-NOIR COLOR DICTIONARY (Time-Dynamic)
// ============================================================

// --- Abyssal Base (static darks — no accent dependency) ---
export const neoBgAbyssalSlate = _fill("bg-slate-950 text-slate-400");
export const neoBgAbyssalZinc = _fill("bg-zinc-950 text-zinc-500");
export const neoBgAbyssalIndigo = _fill("bg-indigo-950");
export const neoBgAbyssalGray = _fill("bg-gray-950");
export const neoBgAbyssalNeutral = _fill("bg-neutral-950");
export const neoBgAbyssalBlack = _fill("bg-black");
export const neoBgAbyssalGradient = _fill(
  "bg-gradient-to-br from-slate-950 via-indigo-950/80 to-black",
);

// --- Off-World Neons (primary + secondary — time-dynamic) ---
export const neoTextNeonCyan = _hourly((p) => p.tx);
export const neoTextNeonCyanBright = _hourly((p) => p.txB);
export const neoTextNeonCyanDeep = _hourly((p) => p.txD);
export const neoTextNeonFuchsia = _hourly((p, s) => s.txD);
export const neoTextNeonFuchsiaBright = _hourly((p, s) => s.tx);
export const neoTextNeonPurple = _hourly((p, s) => s.txD);
export const neoTextNeonIndigo = _hourly((p, s) => s.tx);
export const neoBgNeonCyanOverlay = _hourly((p) => p.neo);
export const neoBorderNeonPink = _hourly((p, s) => `${s.bS} ${s.rS}`);

// --- Warm Refuge (warm palette — time-dynamic) ---
export const neoTextRefugeAmber = _hourly((p, s, w) => w.tx);
export const neoTextRefugeAmberBright = _hourly((p, s, w) => w.txB);
export const neoTextRefugeAmberDeep = _hourly((p, s, w) => w.txD);
export const neoTextRefugeOrange = _hourly((p, s, w) => w.txD);
export const neoTextRefugeOrangeBright = _hourly((p, s, w) => w.tx);
export const neoTextRefugeYellow = _fill("text-yellow-200");
export const neoTextRefugeYellowDeep = _fill("text-yellow-500");
export const neoBgRefugeOrangeOverlay = _hourly((p, s, w) => w.wOv);

// --- System Code / HUD (hud palette — time-dynamic) ---
export const neoTextHudEmerald = _hourly((p, s, w, h) => h.tx);
export const neoTextHudEmeraldBright = _hourly((p, s, w, h) => h.txB);
export const neoTextHudTeal = _hourly((p, s, w, h) => h.tx);
export const neoTextHudTealBright = _hourly((p, s, w, h) => h.txB);
export const neoTextHudLime = _fill("text-lime-400");
export const neoTextHudPurple = _fill("text-purple-300");
export const neoBgHudEmeraldOverlay = _hourly((p, s, w, h) => h.hov);
export const neoBgHudEmeraldGlow = _hourly((p, s, w, h) => h.hovl);

// --- Atmospheric Washes (time-dynamic based on roles) ---
export const neoWashIndigo = _hourly((p) => p.ov);
export const neoWashFuchsia = _hourly((p, s) => s.ovl);
export const neoWashCyan = _hourly((p) => p.ov);
export const neoWashAmber = _hourly((p, s, w) => w.ov);
export const neoWashFogWhite = _fill("bg-white/5");
export const neoWashFogBlack = _fill("bg-black/40");
export const neoWashVolumetricFog = _fill(
  "bg-gradient-to-t from-transparent to-indigo-500/10 mix-blend-screen pointer-events-none",
);

// ============================================================
// II. HOLOGRAPHIC INTERFACES & SYSTEM TYPOGRAPHY
// ============================================================
export const neoTypeSyntax = _fill(
  "font-mono text-[10px] sm:text-xs md:text-sm font-medium tracking-widest uppercase antialiased leading-relaxed",
);
export const neoTypeSignage = _fill(
  "font-sans font-black text-2xl md:text-4xl tracking-tighter uppercase italic drop-shadow-md",
);
export const neoHudTerminalBorder = _hourly(
  (p) =>
    `border ${p.b40} rounded-sm ring-1 ring-inset ${p.r20} divide-y ${p.d20}`,
);
export const neoHudInteractive = _hourly(
  (p) => `${p.hTx} transition-all duration-300 ease-in-out cursor-crosshair`,
);

// ============================================================
// III. STRUCTURAL GLASSMORPHISM & REFLECTIONS (static)
// ============================================================
export const neoGlassWetBase = _fill(
  "backdrop-blur-sm md:backdrop-blur-md bg-white/5 backdrop-brightness-75 backdrop-contrast-125 border border-white/10 shadow-2xl",
);
export const neoGlassWindowPanes = _fill(
  "border-b-4 border-r-2 border-slate-800 border-opacity-80 shadow-2xl shadow-black/50 rounded-sm",
);
export const neoGlassLiquidRain = _fill(
  "mix-blend-overlay opacity-40 pointer-events-none bg-gradient-to-b from-transparent to-slate-900/80",
);
export const neoGlassHarshArchitecture = _fill(
  "border-l-4 border-r-2 border-b border-slate-800 rounded-sm",
);

// ============================================================
// IV. EMISSIVE LIGHTING (primary/secondary palette glows)
// ============================================================
export const neoGlowCyan = _hourly((p) => p.glo);
export const neoGlowMagenta = _hourly((p, s) => s.gloS);
export const neoGlowAmberWash = _fill(
  "shadow-[inset_0_-20px_50px_rgba(245,158,11,0.15)]",
);
export const neoGlowFlicker = _fill(
  "animate-pulse hover:animate-none transition-opacity duration-150 ease-in-out",
);
export const neoGlowTerminalScreen = _fill(
  "opacity-80 hover:opacity-100 shadow-[inset_0_0_20px_rgba(16,185,129,0.2)]",
);

// --- Pre-Assembled Components (primary accent for border/text) ---
export const neoComponentFloatingHUD = _hourly(
  (p) =>
    `backdrop-blur-md bg-white/5 border ${p.b30} font-mono ${p.tx} text-xs shadow-[inset_0_0_20px_rgba(16,185,129,0.2)]`,
);
