"""Link-16 shared network track picture: one common tactical picture per team.

Rather than every platform independently fusing the whole datalinked report set
(O(N) work per receiver -> O(N^2) per tick), a team's contributed reports are fused
and tracked **once** here into shared *system tracks* with stable Track Numbers. Each
platform then reads this picture and reconciles it with its own fresh local radar
tracks (see ``track_reconcile``).

A report joins the net after its **source's net-entry latency** (TDMA-style); after
that every connected member sees it together. Missile seekers are non-cooperative and
never participate — they keep their own onboard seeker.
"""

from __future__ import annotations

from bvr_marl_core.domain.information import FrameReference
from bvr_marl_core.radar.core.data_link import DataLink
from bvr_marl_core.radar.obs.cluster import Clusterer
from bvr_marl_core.radar.tracking.helpers.recenter_logic import transform_states_between_refs_batch
from bvr_marl_core.radar.tracking.tracker import TrackerManager
from bvr_marl_core.simulator.core.helpers import Position

# Network Track Numbers live in a disjoint high range so a platform can merge them
# with its own local track ids without collision (see track_reconcile).
NETWORK_TRACK_ID_BASE = 1_000_000


def network_sensor_id(team) -> tuple[str, object]:
    """Evaluator-only namespace for a team's shared system tracks."""
    return ("network", team)


def _is_range_denied(report) -> bool:
    try:
        return bool(report.get("range_denied", False))
    except AttributeError:
        return bool(getattr(report, "range_denied", False))


class NetworkTrackPicture:
    """Persistent shared fusion+tracking picture for one datalink team."""

    def __init__(
        self,
        team,
        reference: Position,
        *,
        assoc_dist: float,
        confirmation_hits: int = 3,
        max_report_age_s: float = 30.0,
        clusterer=None,
        motion_model: str = "cv",
    ):
        self.team = team
        self.reference = reference
        self.max_report_age_s = float(max_report_age_s)
        self.clusterer = clusterer or Clusterer(10.0, assoc_dist)
        self.tracker = TrackerManager(
            assoc_dist,
            confirmation_hits=confirmation_hits,
            motion_model=motion_model,
        )
        # Shared Track Numbers start in the network range so they never collide with
        # a platform's local (own-radar) track ids.
        self.tracker._next_track_id = NETWORK_TRACK_ID_BASE
        self.tracker.set_export_reference(reference)
        self.tracks: list = []

    def update(self, reports, tick_secs: float, current_time_s: float) -> list:
        """Fuse and track the team's net-entered reports in the shared frame."""
        clusters = DataLink("full").fuse_reports(
            reports,
            self.reference,
            clusterer=self.clusterer,
            cooperative=True,
            current_time_s=current_time_s,
            max_report_age_s=self.max_report_age_s,
        )
        self.tracks = self.tracker.update_tracks(
            clusters,
            tick_secs,
            self.reference,
            missed_increment=max(0.0, float(tick_secs)),
        )
        return self.tracks

    def tracks_in_frame(self, receiver_position: Position) -> list:
        """Project the shared network tracks into a reading platform's ENU frame.

        The transform is a rigid ENU-to-ENU rotation+translation, so the estimate is
        frame-invariant; only its coordinates change to the receiver's frame.
        """
        frame = FrameReference(receiver_position.lat, receiver_position.lon, receiver_position.alt)
        if not self.tracks:
            return []
        states, covariances = transform_states_between_refs_batch(
            [track.state for track in self.tracks],
            [track.covariance for track in self.tracks],
            [self.reference] * len(self.tracks),
            receiver_position,
        )
        return [
            track._with_rigid_frame_transform(
                state,
                covariance,
                frame,
            )
            for track, state, covariance in zip(self.tracks, states, covariances, strict=True)
        ]


def _net_members(sim) -> dict[object, list]:
    """Group full-datalink aircraft (never missiles) by team."""
    teams: dict[object, list] = {}
    for unit in sim.active_units.values():
        if bool(getattr(unit, "is_missile", False)):
            continue
        radar = getattr(unit, "radar", None)
        link = getattr(radar, "data_link", None)
        if link is None or link.get_mode() != "full":
            continue
        teams.setdefault(getattr(unit, "group", None), []).append(unit)
    return teams


def _transmitting_members(sim, members) -> list:
    """Members whose report entered the shared net during this tick."""
    if len(members) < 2:
        return []
    is_up = getattr(sim, "is_datalink_up", None)
    if not callable(is_up):
        return list(members)
    return [
        source
        for source in members
        if any(
            peer is not source and is_up(getattr(source, "id", None), getattr(peer, "id", None))
            for peer in members
        )
    ]


def _gather_net_reports(members) -> list:
    """Union of each member's net contribution, deduped by source-qualified identity.

    A member contributes its current reports plus its recent *jammer* strobe history
    (``_shared_source_detections`` with source == receiver, so the per-distance latency
    term drops out), which lets asynchronously dwelling radars triangulate an emitter
    across ticks. Replayed IRST bearings are deliberately excluded: pooling several ticks
    of passive many-target angle-only reports re-triangulates a temporally-mixed bearing
    set each tick and manufactured ghost intersections. Current-tick IRST bearings still
    triangulate normally within the tick.
    """
    link = DataLink("full")
    seen: set = set()
    gathered: list = []
    for unit in members:
        # Keep each source's jammer strobe history (asynchronously dwelling radars
        # triangulate a single emitter across ticks) but drop replayed IRST bearings:
        # re-triangulating a temporally-mixed set of many-target passive bearings each
        # tick manufactures combinatorial ghost intersections and unstable track births.
        contributed = link.net_entry_reports(unit.radar, unit.position, exclude_irst_history=True)
        for report in contributed or ():
            key = (getattr(report, "source_id", None), getattr(report, "report_id", None))
            if key in seen:
                continue
            seen.add(key)
            gathered.append(report)
    return gathered


def _team_reference(members) -> Position:
    """Stable team ENU origin (centroid of the contributing platforms)."""
    positions = [unit.position for unit in members if getattr(unit, "position", None) is not None]
    if not positions:
        return Position(0.0, 0.0, 0.0)
    count = float(len(positions))
    return Position(
        sum(p.lat for p in positions) / count,
        sum(p.lon for p in positions) / count,
        sum(p.alt for p in positions) / count,
    )


def _tracker_params(members) -> tuple[float, int, float, object, str]:
    """Borrow association/confirmation/age settings from a representative member."""
    for unit in members:
        tracker = getattr(unit.radar, "tracker_manager", None)
        if tracker is not None:
            return (
                float(tracker.assoc_dist),
                int(tracker.confirmation_hits),
                float(getattr(unit.radar, "max_report_age_s", 30.0)),
                getattr(unit.radar, "clusterer", None),
                str(getattr(tracker, "motion_model", "cv")),
            )
    return (5000.0, 3, 30.0, None, "cv")


def update_network_pictures(sim, tick_secs: float) -> dict:
    """Build/update one shared network picture per datalink team for this tick.

    Runs once per tick between the global report-freeze stage and the per-platform
    consume stage. Returns ``sim.network_pictures`` (team -> NetworkTrackPicture).
    """
    if str(getattr(sim, "information_mode", "sensor_limited")).lower() == "oracle":
        sim.network_pictures = {}
        return sim.network_pictures

    pictures = getattr(sim, "network_pictures", None)
    if pictures is None:
        pictures = {}
        sim.network_pictures = pictures

    current_time_s = float(getattr(sim, "elapsed_time_s", 0.0) or 0.0)
    # A shared picture only helps teams with at least two networked platforms; a lone
    # aircraft has no net to read and tracks purely on its own radar.
    teams = {team: members for team, members in _net_members(sim).items() if len(members) >= 2}
    for team, members in teams.items():
        picture = pictures.get(team)
        if picture is None:
            assoc_dist, hits, max_age, clusterer, motion_model = _tracker_params(members)
            picture = NetworkTrackPicture(
                team,
                _team_reference(members),
                assoc_dist=assoc_dist,
                confirmation_hits=hits,
                max_report_age_s=max_age,
                clusterer=clusterer,
                motion_model=motion_model,
            )
            pictures[team] = picture
        picture.update(
            _gather_net_reports(_transmitting_members(sim, members)),
            tick_secs,
            current_time_s,
        )
        refresh_attribution = getattr(sim, "refresh_contact_truth_associations", None)
        if callable(refresh_attribution):
            refresh_attribution(
                network_sensor_id(team),
                {track.track_id: track.report_lineage for track in picture.tracks},
            )

    # Drop pictures for teams that no longer field any networked platform.
    for team in [team for team in pictures if team not in teams]:
        del pictures[team]
    return pictures
