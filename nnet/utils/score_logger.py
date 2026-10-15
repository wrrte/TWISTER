"""Buffer CPU episode statistics and publish each complete episode once."""


class ScoreLogger:
    def __init__(self, writer, wandb_run):
        self.writer = writer
        self.wandb_run = wandb_run
        self.episodes = []
        if wandb_run is not None:
            wandb_run.define_metric("sample/*", step_metric="sample/env_steps")
            wandb_run.define_metric("eval/*", step_metric="eval/model_step")

    def add_episode(self, env_name, reward, frames, action_repeat, env_index,
                    episode, model_step, action_step):
        # All arguments are CPU scalars; this never reads a CUDA tensor.
        self.episodes.append({
            "sample/{}_reward".format(env_name): reward,
            "sample/{}_episode_steps".format(env_name): frames // action_repeat,
            "sample/env_index": env_index,
            "sample/episode": episode,
            "sample/model_step": model_step,
            "sample/action_step": action_step,
            "sample/env_steps": action_step // action_repeat,
        })

    def flush(self):
        for record in self.episodes:
            if self.writer is not None:
                for key, value in record.items():
                    # A separate namespace prevents duplicate wandb history
                    # entries when TensorBoard synchronization is enabled.
                    tag = "Training-episode/" + key.removeprefix("sample/")
                    self.writer.add_scalar(tag, value, record["sample/episode"])
            if self.wandb_run is not None:
                # One call per complete episode, including simultaneous finishes.
                self.wandb_run.log(record)
        self.episodes.clear()

    def log_evaluation(self, metrics, model_step, action_step, action_repeat,
                       episodes, num_envs):
        if self.wandb_run is None or "score" not in metrics:
            return
        self.wandb_run.log({
            "eval/episode_avg_return": float(metrics["score"]),
            "eval/episode_avg_steps": float(metrics["steps"]),
            "eval/model_step": model_step,
            "eval/env_steps": action_step // action_repeat,
            "eval/episodes": episodes,
            "eval/num_envs": num_envs,
        })
