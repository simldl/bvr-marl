"""Progress callback for displaying training progress."""

from ray import tune


class ProgressCallback(tune.Callback):
    """Custom callback to display training progress bar."""

    def __init__(self, total_iterations, start_iteration=0):
        self.total_iterations = total_iterations
        self.start_iteration = start_iteration
        self.last_iteration = start_iteration
        print(f"\nTraining Progress (Target: {total_iterations} iterations)")
        print("=" * 80)

    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called after each trial reports results."""
        current_iter = result.get("training_iteration", 0)

        if current_iter != self.last_iteration:
            self.last_iteration = current_iter
            progress_pct = (current_iter / self.total_iterations) * 100

            # Create progress bar (using ASCII characters for Windows compatibility)
            bar_length = 50
            filled_length = int(bar_length * current_iter // self.total_iterations)
            bar = '#' * filled_length + '-' * (bar_length - filled_length)

            # Get metrics - try multiple possible keys for reward
            # Note: result is a nested dict, not flat with slash keys
            reward = (
                result.get("env_runners", {}).get("episode_return_mean") or
                result.get("episode_reward_mean") or
                result.get("sampler_results", {}).get("episode_reward_mean") or
                0.0
            )
            time_total = result.get("time_total_s", 0)

            print(f"\rProgress: [{bar}] {progress_pct:.1f}% | "
                  f"Iter: {current_iter}/{self.total_iterations} | "
                  f"Reward: {reward:.2f} | "
                  f"Time: {time_total/60:.1f}m", end='', flush=True)

            # Print newline at checkpoints or completion
            if current_iter % 10 == 0 or current_iter >= self.total_iterations:
                print()
