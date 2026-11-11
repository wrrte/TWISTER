"""Buffer CPU episode statistics and publish each complete episode once."""


class ScoreLogger:
    def __init__(self, writer, wandb_run):
        self.writer = writer
        self.wandb_run = wandb_run
        self.episodes = []
        self.scalar_tags = set()
        if wandb_run is not None:
            wandb_run.define_metric("sample/*", step_metric="sample/env_steps")
            wandb_run.define_metric("eval/*", step_metric="eval/model_step")

    def log_scalars(self, tag, scalars, step, extra=None):
        """Publish one scalar record, without syncing any TensorBoard files."""
        if self.wandb_run is None or not scalars:
            return
        step_key = tag + "/global_step"
        if tag not in self.scalar_tags:
            self.wandb_run.define_metric(tag + "/*", step_metric=step_key)
            self.scalar_tags.add(tag)
        record = dict(scalars)
        record[step_key] = int(step)
        if extra is not None:
            record.update(extra)
        self.wandb_run.log(record)

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
            scalars = {}
            for key, value in record.items():
                # Preserve the existing TensorBoard scalar names in W&B too.
                tag = "Training-episode/" + key.removeprefix("sample/")
                scalars[tag] = value
                if self.writer is not None:
                    self.writer.add_scalar(tag, value, record["sample/episode"])
            # One call per complete episode, including simultaneous finishes.
            self.log_scalars("Training-episode", scalars, record["sample/episode"], extra=record)
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
