"""
Missile diagnostic tracker for analyzing guidance accuracy and removal reasons.

Usage:
    from bvr_marl_core.missiles.diagnostics.missile_tracker import MissileDiagnostics

    # In simulation loop:
    diagnostics = MissileDiagnostics()
    diagnostics.track_missile_update(missile, sim)

    # After simulation:
    diagnostics.print_summary()
"""

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class MissileDiagnostics:
    """Tracks missile flight data for post-simulation analysis."""

    def __init__(self):
        self.missile_data: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        self.removal_reasons: dict[str, str] = {}
        self.launch_times: dict[str, float] = {}

    def track_missile_launch(self, missile, sim_time: float):
        """Track when a missile is launched."""
        missile_id = missile.id
        self.launch_times[missile_id] = sim_time

        # Record initial state
        self.missile_data[missile_id]["time"].append(sim_time)
        self.missile_data[missile_id]["position"].append(
            (missile.position.lat, missile.position.lon, missile.position.alt)
        )
        self.missile_data[missile_id]["speed"].append(missile.speed)
        self.missile_data[missile_id]["fuel"].append(missile.engine.fuel_s)

        # Record target info
        if hasattr(missile, "target") and missile.target:
            self.missile_data[missile_id]["target_id"].append(missile.target.id)
        else:
            self.missile_data[missile_id]["target_id"].append(None)

        # Record datalink mode
        if hasattr(missile.radar, "data_link"):
            self.missile_data[missile_id]["datalink_mode"].append(
                missile.radar.data_link.get_mode()
            )

    def track_missile_update(self, missile, sim, sim_time: float):
        """Track missile state during each update."""
        missile_id = missile.id

        if missile_id not in self.launch_times:
            self.track_missile_launch(missile, sim_time)

        self.missile_data[missile_id]["time"].append(sim_time)
        self.missile_data[missile_id]["position"].append(
            (missile.position.lat, missile.position.lon, missile.position.alt)
        )
        self.missile_data[missile_id]["speed"].append(missile.speed)
        self.missile_data[missile_id]["fuel"].append(missile.engine.fuel_s)
        self.missile_data[missile_id]["elapsed_time"].append(missile.elapsed_time_s)

        locked_target = None
        if hasattr(missile.radar, "get_locked_target"):
            try:
                locked_target = missile.radar.get_locked_target()
            except Exception:
                pass
        self.missile_data[missile_id]["locked_target"].append(locked_target)

        if hasattr(missile.radar, "data_link"):
            self.missile_data[missile_id]["datalink_mode"].append(
                missile.radar.data_link.get_mode()
            )

        if hasattr(missile.radar, "get_default_group_radars"):
            try:
                group_radars = missile.radar.get_default_group_radars(sim)
                self.missile_data[missile_id]["group_radar_count"].append(
                    len(group_radars) if group_radars else 0
                )
            except Exception:
                self.missile_data[missile_id]["group_radar_count"].append(0)

        if hasattr(missile, "target_provider"):
            guidance_target = missile.target_provider.get_guidance_target()
            self.missile_data[missile_id]["has_guidance_target"].append(guidance_target is not None)

        removal_reason = self._check_removal_reason(missile, sim)
        if removal_reason:
            self.missile_data[missile_id]["removal_warning"].append(removal_reason)

    def track_missile_removal(self, missile_id: str, reason: str, sim_time: float):
        """Track when and why a missile is removed."""
        self.removal_reasons[missile_id] = reason
        if missile_id in self.missile_data:
            self.missile_data[missile_id]["removal_time"] = sim_time
            self.missile_data[missile_id]["removal_reason"] = reason

    def _check_removal_reason(self, missile, sim) -> str:
        """Check what removal condition would trigger (if any)."""
        if missile.engine.should_remove_missile_on_energy(missile):
            if missile.speed < 50.0:
                return f"low_speed({missile.speed:.1f}m/s)"
            if missile.position.alt < -100.0:
                return f"ground_impact({missile.position.alt:.1f}m)"

        if hasattr(missile, "map_limits") and missile.map_limits:
            if (
                missile.position.lat <= missile.map_limits.bottom_lat
                or missile.position.lat >= missile.map_limits.top_lat
                or missile.position.lon <= missile.map_limits.left_lon
                or missile.position.lon >= missile.map_limits.right_lon
            ):
                return "boundary_violation"

        life_time_s = getattr(missile, "life_time_s", 100.0)
        if missile.elapsed_time_s >= life_time_s:
            return f"lifetime_expired({missile.elapsed_time_s:.1f}s)"

        if missile.engine.fuel_s == 0.0:
            motor_burn_s = getattr(missile, "motor_burn_s", 12.0)
            glide_time = missile.elapsed_time_s - motor_burn_s
            if glide_time > 120.0:
                return f"long_glide({glide_time:.1f}s)"

        if hasattr(missile, "target") and missile.target and missile.engine.fuel_s == 0.0:
            try:
                from bvr_marl_core.simulator.core.helpers import units_distance_km

                dist_km = units_distance_km(missile, missile.target)
                if dist_km > 100.0:
                    return f"too_far({dist_km:.1f}km)"
            except Exception:
                pass

        locked_target = None
        if hasattr(missile.radar, "get_locked_target"):
            try:
                locked_target = missile.radar.get_locked_target()
            except Exception:
                pass

        if locked_target is None and missile.elapsed_time_s > 120.0:
            return f"lost_lock({missile.elapsed_time_s:.1f}s)"

        return None

    def print_summary(self):
        """Print diagnostic summary for all tracked missiles."""
        print("\n" + "=" * 80)
        print("MISSILE DIAGNOSTICS SUMMARY")
        print("=" * 80)

        for missile_id, data in self.missile_data.items():
            print(f"\nMissile: {missile_id}")
            print(f"  Launch time: {data['time'][0]:.1f}s")

            if "removal_time" in data:
                flight_time = data["removal_time"] - data["time"][0]
                print(f"  Flight time: {flight_time:.1f}s")
                print(f"  Removal reason: {data.get('removal_reason', 'unknown')}")
            else:
                print("  Status: Still active")

            # Analyze lock stability
            if "locked_target" in data:
                locked_targets = [t for t in data["locked_target"] if t is not None]
                lock_ratio = (
                    len(locked_targets) / len(data["locked_target"]) if data["locked_target"] else 0
                )
                print(f"  Lock stability: {lock_ratio * 100:.1f}% of flight")

                if lock_ratio < 0.5:
                    print("    WARNING: Poor lock stability!")

            # Analyze datalink usage
            if "datalink_mode" in data:
                modes = data["datalink_mode"]
                mode_changes = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])
                print(f"  Datalink mode changes: {mode_changes}")
                print(f"  Final mode: {modes[-1] if modes else 'unknown'}")

            # Analyze group radar availability
            if "group_radar_count" in data:
                avg_group_radars = sum(data["group_radar_count"]) / len(data["group_radar_count"])
                print(f"  Avg group radars available: {avg_group_radars:.1f}")

                if avg_group_radars < 1.0:
                    print("    WARNING: Low group radar availability!")

            # Analyze guidance target availability
            if "has_guidance_target" in data:
                guidance_ratio = sum(data["has_guidance_target"]) / len(data["has_guidance_target"])
                print(f"  Guidance target availability: {guidance_ratio * 100:.1f}%")

                if guidance_ratio < 0.8:
                    print("    WARNING: Frequent loss of guidance target!")

            # Show removal warnings
            if "removal_warning" in data and data["removal_warning"]:
                unique_warnings = set(data["removal_warning"])
                print(f"  Removal warnings during flight: {', '.join(unique_warnings)}")

        print("\n" + "=" * 80)

        # Overall statistics
        total_missiles = len(self.missile_data)
        removed_missiles = sum(1 for d in self.missile_data.values() if "removal_reason" in d)

        print("\nOVERALL STATISTICS:")
        print(f"  Total missiles tracked: {total_missiles}")
        print(f"  Missiles removed: {removed_missiles}")

        if removed_missiles > 0:
            removal_reasons = defaultdict(int)
            for data in self.missile_data.values():
                if "removal_reason" in data:
                    removal_reasons[data["removal_reason"]] += 1

            print("\n  Removal reasons:")
            for reason, count in sorted(removal_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"    {reason}: {count} ({count / removed_missiles * 100:.1f}%)")

        print("=" * 80 + "\n")
