"""Reward logging functionality."""
import json
import csv
from pathlib import Path
from datetime import datetime


def save_reward_logs(reward_history, output_dir="output/reward_logs"):
    """
    Save detailed reward logs to CSV and JSON files.

    Args:
        reward_history: List of dicts containing reward data per timestep
        output_dir: Directory to save the logs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON (complete structure)
    json_file = output_path / f"rewards_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(reward_history, f, indent=2)
    print(f"Saved JSON reward log to: {json_file}")

    # Save as CSV (flattened for easy analysis)
    csv_file = output_path / f"rewards_{timestamp}.csv"

    if not reward_history:
        print("No reward data to save")
        return

    # Collect all possible reward categories from all agents and timesteps
    all_categories = set()
    for step_data in reward_history:
        for agent_id, agent_data in step_data.items():
            if agent_id not in ['timestep', 'sim_time']:
                if 'breakdown' in agent_data:
                    all_categories.update(agent_data['breakdown'].keys())

    # Create CSV with columns: timestep, sim_time, agent_id, total_reward, category1, category2, ...
    categories_sorted = sorted(all_categories)

    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['timestep', 'sim_time', 'agent_id', 'total_reward'] + categories_sorted
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for step_data in reward_history:
            timestep = step_data.get('timestep', 0)
            sim_time = step_data.get('sim_time', 0.0)

            for agent_id, agent_data in step_data.items():
                if agent_id in ['timestep', 'sim_time']:
                    continue

                row = {
                    'timestep': timestep,
                    'sim_time': sim_time,
                    'agent_id': agent_id,
                    'total_reward': agent_data.get('total_reward', 0.0)
                }

                # Add breakdown categories
                if 'breakdown' in agent_data:
                    for category in categories_sorted:
                        row[category] = agent_data['breakdown'].get(category, 0.0)

                writer.writerow(row)

    print(f"Saved CSV reward log to: {csv_file}")

    # Print summary statistics
    print("\n=== Reward Summary ===")
    agent_totals = {}
    for step_data in reward_history:
        for agent_id, agent_data in step_data.items():
            if agent_id in ['timestep', 'sim_time']:
                continue
            if agent_id not in agent_totals:
                agent_totals[agent_id] = 0.0
            agent_totals[agent_id] += agent_data.get('total_reward', 0.0)

    for agent_id, total in sorted(agent_totals.items()):
        team = "Blue" if agent_id.startswith("A") else "Red"
        print(f"{agent_id} ({team}): {total:+.2f}")
    print(f"Total timesteps: {len(reward_history)}")
    print("=" * 30 + "\n")
