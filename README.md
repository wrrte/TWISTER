# Learning Transformer-based World Models with Contrastive Predictive Coding (TWISTER)

This is the official repository of TWISTER (Transformer-based World model wIth contraSTivE Representations).

**Read TWISTER paper on [OpenReview](https://openreview.net/forum?id=YK9G4Htdew) |
[Arxiv](https://arxiv.org/abs/2503.04416)**

<img src='media/twister.gif' width="100%"/>

## Method

We introduce TWISTER, a Transformer model-based reinforcement learning algorithm using action-conditioned Contrastive Predictive Coding (AC-CPC) to learn high-level feature representations and improve the agent performance. We evaluate our method on the commonly used Atari 100k benchmark and DeepMind Control Suite, demonstrating stronger performance in both discrete and continuous action spaces.

<img src='media/method.png' width="100%"/>

The world model learns feature representations by maximizing the mutual information between model states and future stochastic states obtained from augmented views of image observations. The encoder network converts image observations into stochastic states, from which a decoder network learns to reconstruct images while the masked attention Transformer network predicts next episode continuations, rewards and stochastic states conditioned on selected actions. The actor and critic networks are trained in latent space with imaginary trajectories generated from the world model to select actions maximizing the expected sum of future rewards.

## Installation

Clone GitHub repository and set up environment
```
git clone https://github.com/burchim/TWISTER && cd TWISTER
./install.sh
```

## Training

### Atari100k Benchmark

The agent can be trained on specific tasks using the 'env_name' variable, which defines the training environment. The editable run defaults in `configs/defaults.json` currently select shared warmup followed by retrieval ON/OFF (`Both`). Logs, replay and checkpoints use `callbacks/<run_name>_warmup/<env_name>`, `callbacks/<run_name>_O/<env_name>`, and `callbacks/<run_name>_X/<env_name>`.

```
env_name=atari100k-alien run_name=atari100k python3 main.py
```

### DeepMind Control Suite

We also provide the implementation for training on DeepMind Control tasks.

```
env_name=dmc-Acrobot-swingup run_name=dmc python3 main.py
```

### Run seed

Pass `--seed` to seed Python, NumPy, PyTorch (CPU and CUDA), and the training and
evaluation environments before model initialization and the first episode:

```bash
env_name=atari100k-seaquest run_name=twister_seed42 python3 main.py --seed 42
env_name=dmc-Acrobot-swingup run_name=dmc_seed42 python3 main.py --seed 42
```

Seeds accept integers from 0 through 4294967295. Training environments use
`(seed + environment_index) % (2**31 - 1)`; evaluation uses the next index.
Set `"seed": 42` in [configs/defaults.json](configs/defaults.json) for a persistent
default, or use `override_config='{"seed":42}'`. The priority is `--seed`, then
`override_config`, then the defaults file. The shipped `"seed": null` keeps the
existing behavior when no seed is specified.

`Both` records the resolved seed and restores the shared checkpoint RNG in each
branch. When manually resuming a branch, omit `--seed` or use its saved value.
The saved checkpoint RNG takes precedence over restarting the seed sequence.
The seed controls randomness; CUDA operations can still be nondeterministic.

### GPU job queues

`6_run_twister_queue.sh` and `7_run_twister_queue.sh` run persistent queue workers
configured for GPU 6 and GPU 7, respectively. To use another GPU, edit the number
in `readonly GPU_ID=...` at the top of the desired script and restart that worker.
Each worker automatically selects a queue based on its configured GPU's model.
Run the desired script using your TWISTER Python environment:

```bash
./6_run_twister_queue.sh
# To run another worker on GPU 7, use a separate terminal:
./7_run_twister_queue.sh
```

Each worker sets `CUDA_VISIBLE_DEVICES` and queries that same NVIDIA GPU index
to choose a queue in the TWISTER directory:

| GPU model | Queue file |
| --- | --- |
| RTX 3090 | `job_queue_3090.txt` |
| RTX A6000 | `job_queue_A6000.txt` |
| TITAN RTX | `job_queue_titan.txt` |
| RTX PRO 6000 Blackwell | `job_queue_pro6k.txt` |
| Other models | `job_queue_default.txt` |

Put one complete shell command on each line, including any `env_name`,
`run_name`, `override_config`, and `--seed` settings. Blank lines and lines
starting with `#` are ignored. The previous GPU 6/7 commands are in
`job_queue_3090.txt`; the two Gopher runs have seed-specific run names so their
outputs do not overlap when executed concurrently. Use distinct run names for
additional jobs on the same environment.

Workers of the same GPU model share a queue. `flock` on the matching
`job_queue_*.lock` file protects removing the first command so concurrent workers
do not take the same entry. Commands run sequentially on each worker, from the
TWISTER directory, using the inherited Python environment. A command is removed
before execution; failures are logged and the worker continues without retrying.
Empty queues are checked every 10 seconds. Stop a worker with Ctrl+C.

When adding jobs while workers are running, use the same lock:

```bash
flock job_queue_3090.lock bash -c 'cat >> job_queue_3090.txt' <<'JOBS'
env_name=atari100k-alien run_name=atari100k_seed42 python3 -u main.py --seed 42
JOBS
```

Pass an optional queue file path to use a custom queue on the script's configured GPU:

```bash
./6_run_twister_queue.sh job_queue_custom.txt
./7_run_twister_queue.sh job_queue_custom.txt
```

### Visualize experiments

```
tensorboard --logdir ./callbacks
```

### Override hyperparameters

Edit [configs/defaults.json](configs/defaults.json) to change persistent run defaults.
It currently contains:

```json
{
  "seed": null,
  "retrieval_enabled": "Both",
  "num_envs": 4,
  "final_eval_enabled": true,
  "final_eval_episodes": 100,
  "final_eval_num_envs": 32,
  "retrieval": {
    "warmup_steps": 50000,
    "trigger_mode": "z_score",
    "z_score_threshold": 3.5,
    "batch_size_reduction": "retrieved"
  }
}
```

Use `override_config` only for settings that differ for one launch. Overrides merge
recursively, so specifying one `retrieval` option retains the other file settings:

```bash
env_name=atari100k-seaquest run_name=twister_compare override_config='{"retrieval":{"batch_size_reduction":"anchors"}}' python3 main.py
```

Other model hyperparameters can also be overridden:

```
env_name=atari100k-alien run_name=atari100k override_config='{"num_envs": 4, "epochs": 100, "eval_episode_saving_path": "./videos"}' python3 main.py
```

### Weights & Biases logging

Training enables W&B by default, including commands run by the GPU queue workers.
Before the configuration creates a model or environments, TWISTER calls
`wandb.login(verify=True, force=True)`. Saved credentials (including `~/.netrc`) or
`WANDB_API_KEY` are reused and verified; a terminal prompt appears if credentials
are missing. To enter credentials again at startup, pass `--wandb_relogin`.
For unattended workers, configure credentials before launching jobs or supply
`WANDB_API_KEY` through the environment.

The default project is `nnet`, preserving the previous destination. Override it
with `--wandb_project` or `WANDB_PROJECT`. Select an account/team with
`--wandb_entity` or `WANDB_ENTITY`; otherwise W&B uses its configured default.
CLI options take priority over W&B settings. At run initialization, TWISTER prints
the resolved entity, project, run name, project URL, and run URL. Run names retain
the callback path: `callbacks/<run_name>/<env_name>` (including warmup/branch
suffixes when applicable).

```bash
env_name=atari100k-gopher run_name=gopher_seed3110 python3 -u main.py --seed 3110 --wandb_project twister --wandb_entity YOUR_TEAM
```

TensorBoard training metrics are synchronized to W&B. Episode returns and periodic
evaluation scores are also recorded directly under `sample/*` and `eval/*`.
W&B initialization errors stop training instead of continuing without logging.
Use `--no-wandb` to train with only local TensorBoard logs. Explicit W&B
`offline`/`disabled` modes skip login and clearly report that no upload occurs.
Evaluation and `pass` modes do not log in. Retrieval ON/OFF subprocesses preserve
`--no-wandb` and reuse the parent's credentials without forcing another prompt.

### Final checkpoint evaluation

After successful training, TWISTER always saves the final checkpoint, including
when the last epoch falls between periodic saves. For the default Atari100k budget
this is `checkpoints_epoch_50_step_100000.ckpt`. By default it then evaluates
exactly 100 completed episodes with up to 32 evaluation environments. This uses
the saved weights already in memory, fresh seeded environments, and the existing
evaluation backend; no replay buffer is reloaded. Periodic evaluation settings
(`eval_episodes` and `num_eval_envs`) are independent of these final settings.

W&B logging (enabled by default) records `Evaluation-final/score` (mean episode
return), `Evaluation-final/steps`, episode count, environment count, and model/action
steps in the same training run's history and summary. The summary also records
the checkpoint path. TensorBoard receives the final score and episode length under
`Evaluation-final/0` even with `--no-wandb`. Logging finishes after final evaluation. With retrieval `Both`,
each completed ON/OFF branch receives its own final evaluation; shared warmup does
not.

```bash
env_name=atari100k-gopher run_name=atari100k_seed3110 override_config='{"retrieval_enabled":false}' python3 -u main.py --seed 3110 --wandb
```

Set `"final_eval_enabled": false` in `configs/defaults.json` or override it for one
launch to skip final evaluation:

```bash
env_name=atari100k-gopher run_name=atari100k_seed3110 override_config='{"retrieval_enabled":false,"final_eval_enabled":false}' python3 -u main.py --seed 3110 --wandb
```

`final_eval_episodes` and `final_eval_num_envs` can also be changed in the defaults
file or `override_config`. They must be positive integers. Interrupted or failed
training does not trigger final evaluation; `--mode evaluation` keeps its existing
behavior.

### Training performance

Training groups equal-size image crops for resizing by default. Each image still
samples its own crop with the original `RandomResizedCrop` parameters, rejection
sampling and fallback; positions are not shared between images. Interpolation,
antialiasing, image order and tensor precision are retained.

Replay sampling caches its ordered list of eligible trajectories until an append,
eviction or load changes it. Logging scalars stay on the GPU until a display or
log is due, and contrastive accuracy is computed on the GPU. Distribution argument
checks are disabled locally in TWISTER to avoid synchronizing every distribution;
set `distribution_validate_args` to `true` when debugging invalid inputs.

These changes leave batch sizes, imagination horizon, environment interaction
schedule, optimizer, sampling distributions, AMP/TF32 settings and FP64 epoch
statistics unchanged. Running processes use their already imported code; the next
launch of `./7_run_twister_queue.sh` uses the changes without additional options.

To use the original per-image crop/resize path and enable distribution checks:

```bash
env_name=atari100k-gopher override_config='{"group_contrastive_augments": false, "distribution_validate_args": true}' python3 main.py
```

The replay cache and deferred logging remain active with these overrides.

### Optional Retrieval

The run defaults currently select `"retrieval_enabled": "Both"`. To run only
retrieval ON, use a boolean override from this directory (the shared
`../retrieval.py` must exist):

```bash
python3 -m pip install einops==0.8.1
env_name=atari100k-seaquest run_name=twister_retrieval override_config='{"retrieval_enabled": true, "retrieval": {"context_length": 8, "warmup_steps": 5000}}' python3 main.py
```

Set `"retrieval_enabled": false` for the original training path. Omit the override
to use the run defaults from `configs/defaults.json`. `"retrieval_enabled": "Both"`
shares warmup and then runs retrieval ON and OFF sequentially. Multiple environments finish their current episodes and
pause individually before the shared checkpoint is saved.

```bash
env_name=atari100k-seaquest run_name=twister_compare python3 main.py
```

`retrieval.batch_size_reduction` accepts `none` (keep every original imagination
start), `retrieved`, `anchors`, or `half`. The run file currently selects `retrieved`;
the model fallback is `none`. See [RETRIEVAL.md](RETRIEVAL.md)
for the count formulas, ON/OFF output paths, resume commands, and verification.

## Evaluation

'--mode evaluation' can be used to evaluate agents. The '--load_last' flag will scan the log directory to load the last checkpoint. '--checkpoint' can also be used to load a specific '.ckpt' checkpoint file.

```
env_name=atari100k-alien run_name=atari100k_O override_config='{"retrieval_enabled":true}' python3 main.py --load_last --mode evaluation
```

Set `num_eval_envs` to batch policy/world-model inference for multiple independent
games using one model on the GPU. Atari uses Gym's `AsyncVectorEnv`, with a
dedicated process per emulator and shared memory for observations, as in STORM.
Workers limit OpenCV/PyTorch to one CPU thread. The default is `1`, which retains
the original sequential evaluator.
`eval_episodes` remains the number of completed games included in the mean,
even when it is not a multiple of `num_eval_envs`:

```bash
env_name=atari100k-gopher run_name=atari100k_seed3120 override_config='{"retrieval_enabled":false,"load_replay_buffer_state_dict":false,"eval_episodes":100,"num_eval_envs":8}' python3 -u main.py --seed 3120 --mode evaluation --load_last --verbose_progress_bar 2
```

Start with 4 or 8 environments; speedup depends on episode lengths, CPU capacity
and GPU memory. The progress bar counts completed episodes. Completed Atari lanes
immediately start another episode, keeping the inference batch full. Their latent
state and previous action reset, and stale Transformer KV entries stay masked
until the new episode fills its own context. Rewards include the terminal step;
only the requested number of completed episodes contribute to the mean.
The first environment keeps the original evaluation seed; extra environments
use subsequent seeds. Batched latent sampling consumes RNG differently, so scores
can differ from sequential evaluation even with the same run seed. Keep the same
`num_eval_envs` when comparing runs. `num_envs` still controls training environments.
For DMC, model inference is batched but rendering/stepping stays on the main thread.
For a comparison with the earlier threaded evaluator, set
`"eval_env_backend":"thread"`; Atari defaults to `"process"` when `num_eval_envs > 1`.
Restart an existing evaluation process to use source changes.

## Script options

```
# Args
-c / --config_file           type=str   default="configs/twister.py"    help="Python configuration file containing model hyperparameters"
-m / --mode                  type=str   default="training"              help="Mode: training, evaluation, pass"
-i / --checkpoint            type=str   default=None                    help="Load model from checkpoint name"
--cpu                        action="store_true"                        help="Load model on cpu"
--load_last                  action="store_true"                        help="Load last model checkpoint"
--wandb / --no-wandb          default=True                               help="Enable/disable W&B training logs and startup login"
--wandb_project              type=str   default=None                    help="W&B project (settings/WANDB_PROJECT, then nnet)"
--wandb_entity               type=str   default=None                    help="W&B account/team (settings/WANDB_ENTITY, then account default)"
--wandb_relogin              action="store_true"                        help="Request credentials again at training startup"
--verbose_progress_bar       type=int,  default=1,                      help="Verbose level of progress bar display"

# Training
--saving_period_epoch        type=int   default=1                       help="Model saving every 'n' epochs"
--log_figure_period_step     type=int   default=None                    help="Log figure every 'n' steps"
--log_figure_period_epoch    type=int   default=1                       help="Log figure every 'n' epochs"
--step_log_period            type=int   default=100                     help="Training step log period"
--keep_last_k                type=int,  default=3,                      help="Keep last k checkpoints"

# Eval
--eval_period_epoch          type=int   default=1                       help="Model evaluation every 'n' epochs"
--eval_period_step           type=int   default=None                    help="Model evaluation every 'n' steps"

# Info
--show_dict                  action="store_true"                        help="Show model dict summary"
--show_modules               action="store_true"                        help="Show model named modules"

# Debug
--detect_anomaly             action="store_true"                        help="Enable or disable the autograd anomaly detection"
```

## Citation

If this code or paper is helpful in your research, please use the following citation:

```
@inproceedings{burchilearning,
  title={Learning Transformer-based World Models with Contrastive Predictive Coding},
  author={Burchi, Maxime and Timofte, Radu},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

## Acknowledgments

Official DreamerV3 Implementation: [https://github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)
