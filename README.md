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

`7_run_twister_queue.sh` runs a persistent queue worker. With no arguments, it
uses GPU 7 and automatically selects a queue based on that GPU's model, matching
`STORM-1/7_train.sh`. Start it using your TWISTER Python environment:

```bash
./7_run_twister_queue.sh
# To run another worker on GPU 6, use a separate terminal:
./7_run_twister_queue.sh 6
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

Pass a GPU index to override GPU 7, or specify both a GPU and a queue file:

```bash
./7_run_twister_queue.sh 0
./7_run_twister_queue.sh 7 job_queue_custom.txt
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

## Script options

```
# Args
-c / --config_file           type=str   default="configs/twister.py"    help="Python configuration file containing model hyperparameters"
-m / --mode                  type=str   default="training"              help="Mode: training, evaluation, pass"
-i / --checkpoint            type=str   default=None                    help="Load model from checkpoint name"
--cpu                        action="store_true"                        help="Load model on cpu"
--load_last                  action="store_true"                        help="Load last model checkpoint"
--wandb                      action="store_true",                       help="Initialize wandb logging"
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
