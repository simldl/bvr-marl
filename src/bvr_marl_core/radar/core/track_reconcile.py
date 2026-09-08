"""Reconcile a platform's own-radar tracks with the shared Link-16 network picture.

Where an own track and a network track gate to the same contact, the platform presents
whichever estimate is more informative (smaller position covariance) under the shared
network Track Number. This gives own-sensor primacy in the common case (a fresh own
radar track with no net latency beats the delayed network estimate) while still adopting
the network estimate when it is genuinely better — most importantly when the platform
only has a bearing-only strobe on a jammer that the network has triangulated to a range.
Network-only contacts fill in what the platform cannot see itself; own-only contacts
(not yet on the net, or the platform is off-net) keep their local track id.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

# chi-square(3), 99.9% — same association gate the tracker uses.
_GATE_3D_999 = 16.266


def _position_and_covariance(track) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(track.state, dtype=float)
    covariance = np.asarray(track.covariance, dtype=float)
    return state[:3], covariance[:3, :3]


def reconcile_local_and_network(local_tracks: list, network_tracks: list) -> list:
    """Merge fresh own-radar tracks with the shared network picture (own frame).

    Both inputs are ``TrackSnapshot`` sequences already expressed in the reading
    platform's ENU frame. Returns one contact per target with stable, shared identity.
    """
    if not network_tracks:
        return list(local_tracks)
    if not local_tracks:
        return list(network_tracks)

    local_pc = [_position_and_covariance(track) for track in local_tracks]
    network_pc = [_position_and_covariance(track) for track in network_tracks]
    costs = np.full((len(network_tracks), len(local_tracks)), np.inf, dtype=float)
    for network_index, (network_position, network_covariance) in enumerate(network_pc):
        for local_index, (local_position, local_covariance) in enumerate(local_pc):
            delta = local_position - network_position
            innovation_covariance = local_covariance + network_covariance
            try:
                nis = float(delta @ np.linalg.solve(innovation_covariance, delta))
            except np.linalg.LinAlgError:
                nis = float(delta @ np.linalg.pinv(innovation_covariance) @ delta)
            if nis <= _GATE_3D_999:
                costs[network_index, local_index] = nis

    matched_by_network: dict[int, int] = {}
    if np.isfinite(costs).any():
        assignment_costs = np.where(np.isfinite(costs), costs, _GATE_3D_999 + 1.0)
        network_indices, local_indices = linear_sum_assignment(assignment_costs)
        matched_by_network = {
            int(network_index): int(local_index)
            for network_index, local_index in zip(
                network_indices.tolist(), local_indices.tolist(), strict=True
            )
            if np.isfinite(costs[network_index, local_index])
        }

    used_local = set(matched_by_network.values())
    reconciled = []
    for network_index, network in enumerate(network_tracks):
        _, network_covariance = _position_and_covariance(network)
        best_index = matched_by_network.get(network_index)
        if best_index is None:
            # Contact only the network sees: present the shared network track as-is.
            reconciled.append(network)
        else:
            # Same contact: keep the more informative estimate (own radar range beats
            # net latency; network triangulation beats an own bearing-only strobe),
            # always under the shared network Track Number.
            local_track = local_tracks[best_index]
            local_position_trace = float(np.trace(local_pc[best_index][1]))
            network_position_trace = float(np.trace(network_covariance))
            if local_position_trace <= network_position_trace:
                reconciled.append(
                    replace(
                        local_track,
                        track_id=network.track_id,
                        source_ids=tuple(
                            sorted(set(local_track.source_ids) | set(network.source_ids), key=str)
                        ),
                        report_lineage=tuple(
                            sorted(
                                set(local_track.report_lineage) | set(network.report_lineage),
                                key=lambda item: (str(item[0]), item[1]),
                            )
                        ),
                    )
                )
            else:
                reconciled.append(network)

    # Own-only contacts (locally detected but not yet correlated on the net).
    reconciled.extend(track for index, track in enumerate(local_tracks) if index not in used_local)
    return reconciled
