// polymarket-nvidia-ceo-forecast-01 entry (Node 22, standard library only).
//
// Forecast: for each market, the five quantiles of its Yes price at every minute
// of the next 4 hours, as five quantiles a minute.
//
// Model — short-window empirical quantiles, centred on the last price, blended
// with a Gaussian random walk:
//
//   1. For each market, take its last 6 hours (360 minutes) of prices.
//   2. At anchor horizons h in {1,2,3,5,8,12,20,30,45,60,90,120,180,240},
//      collect the h-step changes p[i+h] - p[i] over the window and take their
//      empirical quantiles at 0.05/0.25/0.5/0.75/0.95. Prices are sticky (about
//      87% of minutes do not move at all), so the empirical quantiles capture
//      what a Gaussian cannot: inner quantiles that sit exactly on the last
//      price at short horizons, and fat-tailed jump risk at long horizons.
//   3. Recentre so the median change is 0: the last price is the best median
//      forecast (prediction-market prices are close to martingales; the
//      empirical median over any window is a worse predictor than the price).
//   4. Blend 50/50 with a Gaussian random walk whose sigma is fitted to the
//      empirical interquartile range at h=240. This regularises the noisy
//      empirical tails (only ~120 overlapping windows inform the h=240
//      quantiles) and keeps the forecast honest when the window was quiet.
//   5. Interpolate the quantile offsets across horizons linearly in sqrt(h).
//   6. Markets with too little history for an anchor fall back to pooled
//      quantiles (same procedure over every market of the case).
//
// Local validation (own scorer, pinball loss, practice cases p00..p05 plus a
// live holdout case v00 cut 2026-11-13T06:00Z):
//   house persistence baseline: practice 0.363589, v00 0.220800
//   this entry (TypeScript, as submitted): practice 0.285746, v00 0.075732
// i.e. about 21% below the baseline on the practice and 66% below on the
// live holdout. Drift ideas (cross-market sum, momentum, mean reversion to
// the 6h/24h mean) were tested and rejected: none improved both sets.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** Anchor horizons (minutes) where empirical quantiles are estimated. */
const ANCHORS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240];
/** How much recent history informs the empirical quantiles (6 hours). */
const WINDOW = 360;
/** Minimum overlapping windows needed to trust a market's own anchor. */
const MIN_SAMPLES = 50;
/** Blend weight of the Gaussian random walk (0 = pure empirical). */
const BLEND = 0.5;
/** Fallback spread (price units per square-root minute) when nothing else applies. */
const FALLBACK_SIGMA = 0.0025;

interface CaseMarket {
  id: string;
  name: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Linear-interpolation quantile of a sorted array, matching numpy's default. */
function quantileSorted(sorted: number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0] as number;
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const a = sorted[lo] as number;
  const b = sorted[hi] as number;
  return a + (b - a) * (pos - lo);
}

/** Empirical quantiles of h-step changes at the anchor horizons. */
function empiricalQuantiles(
  prices: number[],
  pooled: Map<number, number[]>,
): Map<number, number[]> {
  const win = prices.slice(-WINDOW);
  const n = win.length;
  const out = new Map<number, number[]>();
  for (const h of ANCHORS) {
    if (n - h < MIN_SAMPLES) {
      const fb = pooled.get(h);
      if (fb) out.set(h, fb.slice());
      continue;
    }
    const diffs: number[] = [];
    for (let i = 0; i <= n - h - 1; i++) diffs.push((win[i + h] as number) - (win[i] as number));
    diffs.sort((a, b) => a - b);
    out.set(h, QUANTILES.map((q) => quantileSorted(diffs, q)));
  }
  return out;
}

/** Pooled empirical quantiles over every market of the case (fallback). */
function pooledQuantiles(markets: CaseMarket[]): Map<number, number[]> {
  const perAnchor = new Map<number, number[]>();
  for (const h of ANCHORS) perAnchor.set(h, []);
  for (const m of markets) {
    const p = m.prices;
    const n = p.length;
    for (const h of ANCHORS) {
      if (n - h < MIN_SAMPLES) continue;
      const arr = perAnchor.get(h) as number[];
      for (let i = 0; i <= n - h - 1; i += 5) arr.push((p[i + h] as number) - (p[i] as number));
    }
  }
  const out = new Map<number, number[]>();
  for (const h of ANCHORS) {
    const arr = perAnchor.get(h) as number[];
    if (arr.length === 0) continue;
    arr.sort((a, b) => a - b);
    out.set(h, QUANTILES.map((q) => quantileSorted(arr, q)));
  }
  return out;
}

/** Interpolate anchor quantiles across horizons, linear in sqrt(h). */
function interpolate(qmap: Map<number, number[]>, horizon: number): number[][] {
  const hs = [...qmap.keys()].sort((a, b) => a - b);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    if (h <= (hs[0] as number)) {
      rows.push((qmap.get(hs[0] as number) as number[]).slice());
    } else if (h >= (hs[hs.length - 1] as number)) {
      rows.push((qmap.get(hs[hs.length - 1] as number) as number[]).slice());
    } else {
      let i = 1;
      while ((hs[i] as number) < h) i++;
      const h0 = hs[i - 1] as number;
      const h1 = hs[i] as number;
      const w = (Math.sqrt(h) - Math.sqrt(h0)) / (Math.sqrt(h1) - Math.sqrt(h0));
      const a = qmap.get(h0) as number[];
      const b = qmap.get(h1) as number[];
      rows.push(a.map((v, j) => (1 - w) * v + w * (b[j] as number)));
    }
  }
  return rows;
}

/** The forecast of one market: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number, pooled: Map<number, number[]>): number[][] {
  if (m.prices.length === 0) {
    // No history at all: emit a flat 0.5 forecast so the case stays valid.
    return Array.from({ length: horizon }, () => [0.5, 0.5, 0.5, 0.5, 0.5]);
  }
  const x = m.prices[m.prices.length - 1] as number;
  let qmap = empiricalQuantiles(m.prices, pooled);
  if (qmap.size === 0) {
    qmap = new Map(ANCHORS.map((h) => [h, Z.map((z) => z * FALLBACK_SIGMA * Math.sqrt(h))]));
  }
  // Quantile offsets per horizon, recentred so the median change is 0.
  const Q = interpolate(qmap, horizon);
  for (let h = 0; h < horizon; h++) {
    const med = (Q[h] as number[])[2] as number;
    for (let j = 0; j < 5; j++) (Q[h] as number[])[j] = ((Q[h] as number[])[j] as number) - med;
  }
  // Gaussian random walk fitted to the empirical IQR at the longest horizon.
  const q240 = Q[horizon - 1] as number[];
  const iqr = Math.max(((q240[3] as number) - (q240[1] as number)), 1e-6);
  const sigma = iqr / (2 * 0.67449) / Math.sqrt(horizon);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const raw = (Q[h - 1] as number[]).map((v, j) => {
      const g = (Z[j] as number) * sigma * Math.sqrt(h);
      return (1 - BLEND) * v + BLEND * g;
    });
    // Clamp to [0,1], enforce non-decreasing quantiles, round to 6 decimals.
    const q = raw.map((v) => x + v);
    for (let j = 0; j < 5; j++) q[j] = Math.min(1, Math.max(0, q[j] as number));
    for (let j = 1; j < 5; j++) q[j] = Math.max(q[j] as number, q[j - 1] as number);
    rows.push(q.map((v) => price6(v)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  }
  const pooled = pooledQuantiles(c.markets);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes, pooled) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
