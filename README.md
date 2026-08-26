# Lomb

A speech-analytics demo that turns a recorded German conversation into a fluency, accuracy, and complexity report — grounded in the CAF framework (Brand & Götz 2011; Huang & Gráf 2025).

**Live demo:** (added once GitHub Pages is enabled — see below)

This repo currently holds the static frontend only (`index.html`). It's a single self-contained file: no build step, no dependencies. The report on the page renders from one JSON-shaped object via a small `LombAPI.analyze(file)` / `renderReport(data)` pair in the page's own script — everything after that call is a drop-in swap once a real backend exists, without touching the rendering code.

## Running it locally

Just open `index.html` in a browser — it's fully self-contained.

## Deploying

This repo is set up for GitHub Pages: Settings → Pages → Deploy from branch → `main` / `root`. Once enabled, the site is live at the `github.io` URL GitHub assigns, and a custom domain can be attached later from that same settings page.

## Status

Frontend demo only, backend not yet connected. See the project's PRDs for the diagnostic methodology and the planned backend architecture.
