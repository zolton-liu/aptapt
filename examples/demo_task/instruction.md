# Task: arithmetic returns demo

Read `environment/data/prices.csv`. Compute one-period arithmetic returns
`price_t / price_(t-1) - 1` in input row order.

Write `/app/output/results.json` with exactly this structure:

```json
{"returns": [0.1, -0.05]}
```

The first input price has no corresponding return and must be omitted. Values
must be finite JSON numbers rounded to eight decimal places.

