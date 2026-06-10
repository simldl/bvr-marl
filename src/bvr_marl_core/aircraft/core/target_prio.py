import numpy as np


class TrackPrioritySystem:
    def __init__(self, own_aircraft):
        self.own_aircraft = own_aircraft

    def _rel_velocity_towards_own(self, track_state):
        state = track_state[0]
        if len(state) < 6:
            return 0.0

        pos_track = np.array(state[:3])
        vel_track = np.array(state[3:6])
        pos_own = np.zeros(3)
        rel_vec = pos_own - pos_track
        rel_dist = np.linalg.norm(rel_vec)
        if rel_dist == 0:
            return 0.0
        dir_vec = rel_vec / rel_dist
        closing_rate = np.dot(vel_track, dir_vec)
        return closing_rate

    def _relative_height(self, track_state):
        state = track_state[0]
        rel_h = -state[2]
        return rel_h

    def _distance(self, track_state):
        state = track_state[0]
        return np.linalg.norm(state[:3])

    def score(self, track_state):
        """
        Compute threat priority score (HIGHER = MORE THREATENING).
        """
        state_vec, cov = track_state
        try:
            state = np.asarray(state_vec, dtype=float).reshape(-1)
            if state.size < 3 or not np.all(np.isfinite(state[:3])):
                return -1e9  # push invalid tracks to the bottom

            dist = float(np.linalg.norm(state[:3]))
            closing = float(self._rel_velocity_towards_own((state, cov)))
            height_diff = float(abs(self._relative_height((state, cov))))

            dist_norm = dist / 50000.0
            closing_norm = closing / 300.0
            height_norm = height_diff / 5000.0

            score = (-1.0 * dist_norm) + (+1.5 * closing_norm) + (-0.8 * height_norm)
            if not np.isfinite(score):
                score = -1e9
            return score
        except Exception:
            return -1e9

    def prioritize(self, track_states):
        """Sort tracks by priority (DESCENDING: highest threat first)."""
        tracks_with_score = []
        for state_vec, cov in track_states:
            sc = self.score((state_vec, cov))
            tracks_with_score.append((state_vec, sc))
        # Sort DESCENDING - highest score (most threatening) first
        tracks_with_score.sort(key=lambda x: x[1], reverse=True)
        return tracks_with_score
