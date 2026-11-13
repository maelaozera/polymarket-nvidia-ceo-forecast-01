# Polymarket NVIDIA CEO Forecast

Forecast, minute by minute and 4 hours ahead, the Yes price of every trading
market of Polymarket's "Next CEO of NVIDIA?" event, as five quantiles
(5/25/50/75/95%) per minute, scored by mean pinball loss.

## Program

`main.ts` (Node 22, standard library only). Run:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

It reads every `<id>.in` in the input directory and writes `<id>.out` with
`{"forecasts":[{"id","q":[[q05,q25,q50,q75,q95], ... 240 rows]}]}`.

## Model

Short-window empirical quantiles, centred on the last price, blended with a
Gaussian random walk:

1. For each market, take its last 6 hours (360 minutes) of prices.
2. At anchor horizons 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240
   minutes, collect the h-step changes and take their empirical quantiles.
   Prices are sticky (~87% of minutes do not move), so the empirical
   quantiles capture what a Gaussian cannot: inner quantiles sitting exactly
   on the last price at short horizons, and fat-tailed jump risk at long
   horizons.
3. Recentre so the median change is 0 — the last price is the best median
   forecast (prediction-market prices are close to martingales).
4. Blend 50/50 with a Gaussian random walk whose sigma is fitted to the
   empirical interquartile range at h=240, to regularise the noisy tails.
5. Interpolate quantile offsets across horizons linearly in sqrt(h).
6. Markets with too little history for an anchor fall back to pooled
   quantiles over every market of the case.

## Local validation

Own scorer (pinball loss, same formula as the rules) on the kit's practice
cases p00–p05 plus a live holdout case v00 (checkpoint 2026-11-13T06:00Z,
answers fetched from Polymarket's public price history):

| model | practice p00–p05 | live holdout v00 |
|---|---|---|
| house persistence baseline | 0.363589 | 0.220800 |
| this entry (TypeScript, as submitted) | 0.285746 | 0.075732 |

Drift ideas (cross-market sum, momentum, mean reversion to the 6h/24h mean)
were tested and rejected: none improved both the practice and the holdout.
