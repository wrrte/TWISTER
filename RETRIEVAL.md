# Retrieval Training

## Scope and Baseline Equivalence

`retrieval_enabled` defaults to the JSON boolean `false`. When disabled, no
retrieval controller, manager, projection, replay index, metadata collation,
extra encoder/value forward, or weighted reduction runs. Network initialization,
replay sampling calls, world-model losses, imagination starts, actor/critic
reductions, optimizer ordering, and target updates follow the original code.
No retrieval tensors enter the model state dict. This preserves both the
mathematical update and the random-number sequence, given identical inputs,
settings, and a deterministic runtime.

When enabled, retrieval changes the actor/critic training distribution. The world
model still receives the original uniform replay batch and its original losses.
This is equality of the world-model update for a fixed batch and RNG state;
future batches and losses can change once the learned policy changes.

## Data and Time Alignment

TWISTER stores overlapping length-`L` windows, not complete episodes. Its
`buffer_capacity` counts windows. Each saved row is
`(observation_t, action_(t-1), reward_t, done_t, is_first_t, model_step_t)`.
Consequently, the trigger residual is

```text
delta_t = reward_(t+1) + gamma * (1 - done_(t+1)) * V_(t+1) - V_t
value_diff_t = V_t - V_(t-1)
anchor_time = t + anchor_offset
```

Values are computed from the already detached world-model posterior features
after the world-model optimizer step. The features are not recomputed. Evaluation
mode and `no_grad` prevent an additional learning path or dropout here.
The shared manager receives shifted rewards/dones and an optional transition
mask. Terminal-to-reset edges, reset states, unavailable contexts, and anchors
across episode boundaries cannot trigger or contribute to the EMA statistics.

The sparse replay view gives each environment monotonically increasing frame
IDs, shared by overlapping windows. It holds references to existing tensors;
it does not allocate another image ring buffer. Reference counts discard frames
only after their last window is evicted. IDs are not reused. Episode boundaries
are checked using both `done` and `is_first`, including time-limit resets.
A context may end at a terminal observation but cannot cross one.

New frames enter a pending index when environment interaction completes a replay
window. Hashes are synchronized before the next trigger/retrieval operation,
including during warmup. Single-frame probability hashing computes encoder
logits directly, uses the encoder's uniform mixture, and consumes no sampling
RNG. Shared normalized images in `[0, 1]` are transferred to the model device
and shifted to TWISTER's `[-0.5, 0.5]` range.

## Imagination and Weights

By default, all `N = batch_size * L` original imagination starts remain present. Each
retrieved context contributes exactly one additional start, its final posterior.
The encoder and TSSM warm up that context in evaluation mode under `no_grad`.
The final stochastic/deterministic state, each layer's K/V cache, the current
`is_first`, the history boundary mask, and the endpoint's real `done` are appended
together. Short caches are left padded to `att_context_left` in the same way as
the original world-model flattening. `observe` receives cloned actions because
it modifies its input in place.

The original continuation/discount weights `c_(i,h)` remain separate from
retrieval sample weights `w_i`. If `ell_(i,h)` already contains those continuation
weights, both actor (including entropy) and critic (including target regularizer)
use this reduction:

```text
loss = sum_i [w_i * mean_h(ell_(i,h))] / sum_i(w_i)
```

Original starts have weight 1. The shared manager distributes total weight 1
over each retrieved anchor group: `anchor_weight` for the anchor, the remaining
mass split between its matches, or weight 1 for a singleton anchor. Multiplying
by weights followed by an ordinary batch mean would instead shrink the update
as the number of matches increases, so that proposal is not used.

`retrieval.batch_size_reduction` controls how many original starts remain. Let
`R` be the number of retrieved contexts and `A` the number of valid anchor groups:

| Mode | Original starts retained |
| --- | --- |
| `none` (default) | `N` |
| `retrieved` | `max(0, N - R)` |
| `anchors` | `max(0, N - A)` |
| `half` | `max(0, N - floor((R + A) / 2))` |

When fewer originals are needed, they are sampled uniformly without replacement
from the existing `batch_size * L` starts, retaining their relative order. The
posterior, every K/V cache, masks, and terminal flag are selected together. World
model batch size and losses are unchanged. Empty retrieval and warmup retain all
original starts. If `R > N`, the original count is zero and all `R` retrieved
contexts remain. This matches STORM/Drama's count formulas at TWISTER's
imagination-start level, rather than reducing the world-model replay batch.

The default remains additive. With default `N=1024` and at most 10 groups,
retrieval has at most `10/1034`, approximately 0.97%, of the nominal sample mass.
Changing `max_contexts` only caps the number of sequences; it does not increase
the total mass of a fixed number of groups. Discount weights can further reduce
their effective gradient contribution. This is an experimental design choice,
not a claim of equivalent retrieval strength across the three algorithms.
TWISTER's return-percentile normalization still uses the combined, unweighted
return population, as in its original algorithm; only loss reduction is weighted.

## Configuration

Pass settings through `override_config` in `configs/twister.py`:

```bash
env_name=atari100k-seaquest run_name=twister_retrieval override_config='{"retrieval_enabled": true, "retrieval": {"context_length": 8, "warmup_steps": 5000, "trigger_mode": "z_score"}}' python3 main.py
```

| Setting in `retrieval` | Default | Meaning |
| --- | --- | --- |
| `context_length` | 8 | Retrieved history length, 1 through `L-2` |
| `batch_size_reduction` | `none` | `none`, `retrieved`, `anchors`, or `half`; counts imagination starts |
| `warmup_steps` | 5000 | Environment decisions summed over environments |
| `max_anchors` / `max_contexts` | 10 / 256 | Groups processed / sequences appended per update |
| `multiplier` / `target` | 5 / 5 | Candidate sampling multiplier / group size target |
| `trigger_mode` / `threshold` | `absolute` / 1.0 | Absolute TD-error trigger |
| `z_score_threshold` / `ema_alpha` | 2.0 / 0.01 | Shared z-score trigger parameters |
| `anchor_offset` / `anchor_weight` | -2 / 0.5 | Anchor time offset / weight within its group |
| `hash_bits` / `hash_sample_mode` | 12 / `probs` | Hash size / `probs`, `mode`, or `sample` |
| `max_bucket_size` | 512 | Maximum entries per bucket |
| `use_pca` / `max_pca_samples` | true / 10000 | Projection fitting / bounded fitting sample |
| `chunk_size` | 256 | Maximum encoder frames per hash chunk |
| `global_rebuild_enable` | true | Enable drift-triggered rebuilding |
| `global_rebuild_threshold` | 0.2 | Rebuild when lazy hit rate is below this |
| `global_rebuild_cooldown` | 2000 | Minimum environment decisions between rebuilds |

The activation flag accepts JSON booleans and the string `"Both"`.
Dynamic warmup (`-1`), named experiment lists, and selectable `value_signal` /
`score_combination` are not implemented. Unsupported settings are rejected.

`action_step` counts repeated emulator frames, so retrieval uses
`action_step // env.action_repeat`. This counts decisions across all environments,
including random prefill, and excludes extra reset rows. Warmup accumulates
trigger statistics but adds no anchors or imagination contexts. At warmup exit
the index is rebuilt even if drift rebuilding is disabled. Later low-hit queries
can rebuild after cooldown; an empty queue alone does not trigger rebuilds.
Rebuilding clears queued anchors because their keys refer to the previous
projection. PCA fitting is bounded by `max_pca_samples`; all-frame hashing is
chunked, without retaining the full replay's latent matrix on the GPU.

## Shared Warmup with `Both`

From this directory, in a Python environment with TWISTER dependencies and ROMs:

```bash
env_name=atari100k-seaquest run_name=twister_compare override_config='{"retrieval_enabled":"Both","num_envs":4,"retrieval":{"warmup_steps":50000,"trigger_mode":"z_score","z_score_threshold":3.5,"batch_size_reduction":"retrieved"}}' python main.py
```

The warmup target counts actual environment decisions summed across environments.
After reaching it, each environment finishes its current episode and then pauses;
time-limit endings count, but life loss alone does not. No further decisions or
replay transitions come from paused environments. Other environments continue,
and training continues with retrieval augmentation disabled, until all are paused.
The actual shared warmup can therefore exceed `warmup_steps`. A target of zero
branches from the initial boundary, before prefill.

The shared checkpoint contains model parameters, optimizer/scaler state, return
normalization, retrieval statistics, replay metadata, and Python/NumPy/Torch RNG.
The process becomes the existing shared branch supervisor, which runs ON then
OFF sequentially, each in a fresh process:

```text
callbacks/twister_compare_warmup/<env>/shared_warmup/  checkpoint and run.json
callbacks/twister_compare_O/<env>/                   retrieval ON
callbacks/twister_compare_X/<env>/                   retrieval OFF
```

Both branches load the same checkpoint and RNG state and restart environments
using the same saved per-environment reset seeds. Simulator state is not captured;
this is a fresh episode start, not an exact continuation of the prior simulator.
Each branch trains only the remaining model updates in `epochs * epoch_length`,
including a partial first epoch when needed. The warmup is not repeated. If the
training budget ends before all environments finish, no branches are launched.
These runs require `load_replay_buffer_state_dict=true` and `accumulated_steps=1`.

Shared replay trajectory files are referenced from the warmup directory; each
branch writes new trajectories to its own directory. Keep the warmup directory
available when resuming branch checkpoints. Replay source files are recorded by
absolute path, so preserve their original paths. The supervisor writes
`branches.json` and `branch_results.json` in `shared_warmup`, and stops the sequence
if a branch fails or is interrupted. Both child completion checkpoints are saved
even if no periodic checkpoint falls on their final update.

To rerun only one branch from a shared checkpoint:

```bash
python main.py --shared_warmup callbacks/twister_compare_warmup/atari100k-seaquest/shared_warmup --retrieval_branch off
```

This restores the saved environment name, model overrides, and original run name,
and writes to the corresponding `_O` or `_X` directory. To continue an interrupted
child instead, use its boolean `retrieval_enabled` setting, its suffixed run name,
the same model settings, and `--load_last` as in ordinary TWISTER training.

TensorBoard metrics under `Training-step/` include `retrieval_original_starts`
and `retrieval_imagination_starts`, in addition to contexts, anchors, hit rate,
warmup status, and weight sum.

## Checkpoints

Enabled checkpoints save the manager's projection, buckets, statistics and
anchors, last rebuild step, and replay window-to-frame metadata. Loading restores
the statistical state and rebuilds the index on the first active retrieval step.
Changing `num_envs` when restoring indexed replay is rejected.
The saved hash projection must also match the configured `hash_bits` and latent
dimension; incompatible projections are rejected with an explicit error.

TWISTER does not save the simulator or unfinished replay streams. On indexed
resume those streams are discarded and new frame IDs are separated by a gap,
so new data cannot become false continuations of old episodes. This provides
consistent retrieval resume, not bit-exact simulator continuation.

Legacy checkpoints have no environment/frame metadata. Each stored window is
therefore treated as an independent segment in retrieval environment 0; unknown
continuity is never inferred. This migration can represent overlapping historical
frames more than once and combines their trigger statistics. Fresh enabled
runs and enabled checkpoints preserve environment IDs and deduplication.
Original replay trajectory files are still required for loading, as before.

## Verification

`tests/test_twister_retrieval.py` runs real TWISTER networks and Adam updates with
a deterministic substitute for the simulator. It compares the disabled path to
the original TWISTER class from nested-repository revision `207843cb645f`, including losses,
parameters, optimizer states, and Python/NumPy/Torch RNG over two updates.
It also covers enabled world-model equality, retrieval augmentation, discrete
and continuous actions, cache/mask equivalence for short/equal/long contexts,
terminal discounts, weighted loss gradients, boundaries, eviction, PCA rebuilds,
warmup, empty retrieval, and enabled/legacy checkpoint resume.

```bash
# From the parent workspace, using TWISTER's Python environment:
python -m unittest discover -s tests -p 'test_twister*.py' -v
```

The branch tests also execute the real `main.py` and process supervisor with
staggered toy episodes, verifying ON/OFF completion from a partial warmup epoch,
shared replay loading from separate directories, and RNG/optimizer restoration.

The shared STORM/Drama retrieval regression suite should also be run in their
Python environment. Full Atari/DMC learning curves and CUDA numerical behavior
are not established by the CPU tests.
