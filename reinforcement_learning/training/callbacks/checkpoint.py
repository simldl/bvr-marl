"""Smart checkpoint management callback."""

import os
import shutil
import glob
from ray import tune


class SmartCheckpointCallback(tune.Callback):
    """
    Custom checkpoint management callback.

    Keeps:
    - 20 milestone checkpoints (one every 5% of training)
    - 2-3 recent backup checkpoints (most recent time-based saves)

    This ensures you have both progress milestones and recent recovery points.
    """

    def __init__(self, total_iterations, checkpoint_dir):
        import time
        self.total_iterations = total_iterations
        self.checkpoint_dir = checkpoint_dir
        self.milestone_checkpoints = set()  # Track milestone checkpoint iterations
        self.all_checkpoints = []  # Track all checkpoints (iteration, path)
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval_seconds = 30 * 60  # 30 minutes

        # Calculate milestone iterations (every 5%)
        self.milestones = set()
        for pct in range(5, 101, 5):  # 5%, 10%, 15%, ..., 100%
            milestone_iter = int((pct / 100.0) * total_iterations)
            if milestone_iter > 0:
                self.milestones.add(milestone_iter)

        print(f"\nCheckpoint management strategy:")
        milestone_samples = sorted(list(self.milestones)[:5])
        print(f"  - Milestone iterations: {milestone_samples}... (every 5%, total: {len(self.milestones)})")
        print(f"  - Will keep: ALL milestones + last 3 non-milestone backups")
        print(f"  - Time-based backups noted every 30 minutes (for monitoring only)")

    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called after each trial reports results - manage checkpoints here."""
        del iteration, trials, info  # Unused but required by callback signature
        import time

        current_iter = result.get("training_iteration", 0)
        current_time = time.time()

        # Find checkpoint directory for this trial
        # Ray saves checkpoints in: storage_path/experiment_name/trial_name/checkpoint_XXXXXX
        if hasattr(trial, 'local_path'):
            trial_dir = trial.local_path
        elif hasattr(trial, 'logdir'):
            trial_dir = trial.logdir
        else:
            # Can't find trial directory, skip checkpoint management
            return

        # Find all checkpoint directories
        checkpoint_pattern = os.path.join(trial_dir, "checkpoint_*")
        checkpoint_dirs = glob.glob(checkpoint_pattern)

        if not checkpoint_dirs:
            return

        # Parse checkpoint iterations from directory names
        checkpoint_map = {}  # iteration -> path
        for cp_path in checkpoint_dirs:
            # Extract iteration from "checkpoint_XXXXXX"
            try:
                cp_name = os.path.basename(cp_path)
                if cp_name.startswith("checkpoint_"):
                    iter_str = cp_name.replace("checkpoint_", "").lstrip("0") or "0"
                    cp_iter = int(iter_str)
                    checkpoint_map[cp_iter] = cp_path
            except (ValueError, IndexError):
                continue

        # Update our tracking
        for cp_iter, cp_path in checkpoint_map.items():
            if (cp_iter, cp_path) not in self.all_checkpoints:
                self.all_checkpoints.append((cp_iter, cp_path))

        # Check if current iteration is a milestone
        if current_iter in self.milestones and current_iter not in self.milestone_checkpoints:
            self.milestone_checkpoints.add(current_iter)
            progress_pct = 100 * current_iter / self.total_iterations
            print(f"\n[Checkpoint] Milestone saved at iteration {current_iter}/{self.total_iterations} "
                  f"({progress_pct:.0f}%) - Will be preserved")

        # Check if we should note a time-based backup
        time_since_last = current_time - self.last_checkpoint_time
        if time_since_last >= self.checkpoint_interval_seconds:
            self.last_checkpoint_time = current_time
            print(f"\n[Checkpoint] Time marker at iteration {current_iter} (30min elapsed)")

        # Clean up old checkpoints
        # Keep: milestone checkpoints + last 3 non-milestone checkpoints
        checkpoints_to_keep = set()

        # 1. Add all milestone checkpoint iterations
        checkpoints_to_keep.update(self.milestone_checkpoints)

        # 2. Add last 3 non-milestone checkpoint iterations
        all_iters = sorted(checkpoint_map.keys())
        non_milestone_iters = [it for it in all_iters if it not in self.milestone_checkpoints]
        recent_backups = non_milestone_iters[-3:] if non_milestone_iters else []
        checkpoints_to_keep.update(recent_backups)

        # 3. Delete checkpoints not in the keep set
        deleted_count = 0
        for cp_iter, cp_path in list(checkpoint_map.items()):
            if cp_iter not in checkpoints_to_keep:
                if os.path.exists(cp_path):
                    try:
                        shutil.rmtree(cp_path)
                        deleted_count += 1
                        # Remove from tracking
                        self.all_checkpoints = [(it, path) for it, path in self.all_checkpoints
                                                if path != cp_path]
                    except Exception as e:
                        print(f"\n[Checkpoint] Warning: Could not delete checkpoint {cp_iter}: {e}")

        # Summary of checkpoint status
        if deleted_count > 0:
            kept_milestone_count = len([it for it in checkpoints_to_keep if it in self.milestone_checkpoints])
            kept_backup_count = len([it for it in checkpoints_to_keep if it not in self.milestone_checkpoints])
            print(f"[Checkpoint] Deleted {deleted_count} old checkpoint(s). "
                  f"Keeping: {kept_milestone_count} milestone(s) + {kept_backup_count} backup(s)")
