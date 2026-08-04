# Adaptive Agent Lab dashboard

This vinext application replays warehouse trajectories and compares operational
metrics for the six Adaptive Agent Lab treatments. It reads
`public/demo-data.json`, which is generated from real simulator transitions by
the Python package.

The bundled data is a **non-confirmatory demonstration**. Learners receive small
convenience training budgets on the displayed scenario, so the dashboard is
useful for inspecting behavior and testing the replay UI—not for claiming an
algorithm ranking or a held-out benchmark result.

## Local development

Requires Node.js `>=22.13.0` and pnpm.

```bash
pnpm install
pnpm dev
```

From the repository root, regenerate the displayed trajectories with:

```bash
aal export-demo \
  --scenario scenarios/small/dynamic-demo.json \
  --output web/public/demo-data.json
```

## Checks

```bash
pnpm lint
pnpm test
pnpm build
```

`pnpm test` includes a production build before checking the rendered HTML.
