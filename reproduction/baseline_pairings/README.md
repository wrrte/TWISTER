Each `<game>/seed<full_FLASH_seed>.json` records one accepted full FLASH result
paired with an original 999N baseline from `results/reusable_baselines.json`.
Workers create these records atomically before marking the pair complete.

The baseline seed, score and original run IDs remain distinct from the full
FLASH seed. A baseline seed may occur in only one pairing per game. Gopher is
excluded. Existing records survive edits to the pool and are reused after an
interruption, so editing the pool only changes future allocations.

Daily cluster sync publishes the current server's records alongside its report.
Both results exporters read this directory automatically. Keep these files when
moving numeric history between servers; they contain no checkpoints or replay.
