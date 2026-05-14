# External Baseline Overlays

The repository intentionally ignores `external/` because baseline source trees,
downloaded datasets, and generated caches can be large and dependency-specific.

Tracked files under `patches/external_overlays/` are small compatibility overlays
that should be copied into the corresponding external baseline checkout before
running the paper baseline wrappers. Use:

```bash
python scripts/setup/bootstrap_external_baselines.py --apply-overlays-only
```

or let the same script clone/copy the external repositories first. Current
overlays include:

- `scGPT-main`: flash-attention compatibility helpers used by the scGPT wrappers.
