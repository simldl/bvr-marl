import numpy as np

class RadarParamPolicy:
    """
    Dynamic policy for calculating radar parameters depending on the mode.
    """
    def __init__(
        self,
        search_h_fov_deg: float,
        search_v_fov_deg: float,
        search_gain_db: float,
        search_snr_db: float,
        search_false_alarm_rate: float = 0.01,
        search_update_rate_hz: float = 1.0,
        min_h_fov_deg: float = 2.0,
        min_v_fov_deg: float = 1.0,
        max_gain_db: float = 50.0,
        min_snr_db: float = 3.0
    ):
        self.search_h_fov_deg = search_h_fov_deg
        self.search_v_fov_deg = search_v_fov_deg
        self.search_gain_db = search_gain_db
        self.search_snr_db = search_snr_db
        self.search_false_alarm_rate = search_false_alarm_rate
        self.search_update_rate_hz = search_update_rate_hz
        self.min_h_fov_deg = min_h_fov_deg
        self.min_v_fov_deg = min_v_fov_deg
        self.max_gain_db = max_gain_db
        self.min_snr_db = min_snr_db

    def get_params_for_mode(
        self, 
        mode: str, 
        lock_h_fov_deg: float = None, 
        lock_v_fov_deg: float = None, 
        gain_scaling: float = None,
        snr_scaling: float = None,
        false_alarm_rate: float = None,
        update_rate_hz: float = None
    ) -> dict:

        if mode == "search":
            return dict(
                h_fov_deg=self.search_h_fov_deg,
                v_fov_deg=self.search_v_fov_deg,
                antenna_gain_db=self.search_gain_db,
                snr_threshold_db=self.search_snr_db,
                false_alarm_rate=self.search_false_alarm_rate,
                update_rate_hz=self.search_update_rate_hz
            )

        elif mode == "track" or mode == "lock":
            # Default: FOV is 10% of the search-FOVs
            h_fov_deg = lock_h_fov_deg if lock_h_fov_deg is not None else max(self.min_h_fov_deg, 0.1 * self.search_h_fov_deg)
            v_fov_deg = lock_v_fov_deg if lock_v_fov_deg is not None else max(self.min_v_fov_deg, 0.1 * self.search_v_fov_deg)
            gain_increase = 10.0 * np.log10(self.search_h_fov_deg / h_fov_deg) if gain_scaling is None else gain_scaling
            antenna_gain_db = min(self.max_gain_db, self.search_gain_db + gain_increase)
            snr_increase = 2.0 if snr_scaling is None else snr_scaling
            snr_threshold_db = max(self.min_snr_db, self.search_snr_db + snr_increase)
            lock_false_alarm = false_alarm_rate if false_alarm_rate is not None else self.search_false_alarm_rate * 0.2
            lock_update_rate_hz = update_rate_hz if update_rate_hz is not None else self.search_update_rate_hz * 2.0

            return dict(
                h_fov_deg=h_fov_deg,
                v_fov_deg=v_fov_deg,
                antenna_gain_db=antenna_gain_db,
                snr_threshold_db=snr_threshold_db,
                false_alarm_rate=lock_false_alarm,
                update_rate_hz=lock_update_rate_hz
            )
        else:
            raise ValueError(f"Unknown radar mode: {mode}")
        
    @staticmethod
    def get_adaptive_fov(distance_m, lock_distance=15000, search_fov=120.0, lock_fov=10.0):
        if distance_m > 2 * lock_distance:
            return search_fov
        elif distance_m > lock_distance:
            alpha = (distance_m - lock_distance) / lock_distance
            return lock_fov + alpha * (search_fov - lock_fov)
        else:
            return lock_fov

    def get_params_for_mode_auto(
        self,
        distance_m: float,
        tracker_stable: bool,
        lock_distance: float = 15000,
        search_fov: float = None,
        lock_fov: float = None,
        update_rate_hz: float = None
    ):

        search_fov = search_fov or self.search_h_fov_deg
        lock_fov = lock_fov or max(2.0, 0.08 * search_fov)


        fov = self.get_adaptive_fov(distance_m, lock_distance=lock_distance, search_fov=search_fov, lock_fov=lock_fov)
        gain_increase = 10 * np.log10(search_fov / fov)
        antenna_gain_db = self.search_gain_db + gain_increase
        snr_threshold_db = self.search_snr_db + (2.0 if tracker_stable else 0.0)
        upd_rate = self.search_update_rate_hz * (2.0 if (tracker_stable or distance_m < lock_distance) else 1.0)

        return dict(
            h_fov_deg=fov,
            v_fov_deg=fov / 2,
            antenna_gain_db=antenna_gain_db,
            snr_threshold_db=snr_threshold_db,
            false_alarm_rate=self.search_false_alarm_rate * (0.2 if tracker_stable else 1.0),
            update_rate_hz=upd_rate
        )


    def __repr__(self):
        return f"<RadarParamPolicy search_h_fov={self.search_h_fov_deg} search_gain={self.search_gain_db}dB search_snr={self.search_snr_db}dB>"


    def update_dynamic_params(self, distance_m=None, tracker_stable=False):

        mode = self.lock_ctrl.get_mode()
        if mode == "auto" and distance_m is not None:
            params = self.get_params_for_mode_auto(
                distance_m=distance_m,
                tracker_stable=tracker_stable
            )
        else:
            params = self.get_params_for_mode(mode)

        self.h_fov_deg = params["h_fov_deg"]
        self.v_fov_deg = params["v_fov_deg"]
        self.antenna_gain_db = params["antenna_gain_db"]
        self.snr_threshold_db = params["snr_threshold_db"]
        self.false_alarm_rate = params["false_alarm_rate"]
        self.update_rate_hz = params["update_rate_hz"]