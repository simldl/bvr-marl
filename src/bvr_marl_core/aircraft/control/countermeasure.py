import logging

from bvr_marl_core.aircraft.control.countermeasure_objects import create_countermeasure

logger = logging.getLogger(__name__)


class AircraftCountermeasureSystem:
    """
    Aircraft countermeasure system that deploys physical countermeasure objects.

    Countermeasures persist in the simulation for several seconds and can affect
    missile seekers during their lifetime.
    """

    def __init__(self, parent):
        self.parent = parent
        self.flares = getattr(parent, "flares", 0)
        self.chaff = getattr(parent, "chaff", 0)
        self.ecm = getattr(parent, "ecm", 0)
        self.decoys = getattr(parent, "decoys", 0)

        self.active_countermeasures = []

        self.flare_params = {
            "lifetime_s": 5.0,
            "effectiveness": 1.0,
            "decay_rate": 0.3,
            "ir_signature": 1000.0,
        }

        self.chaff_params = {
            "lifetime_s": 8.0,
            "effectiveness": 1.0,
            "decay_rate": 0.15,
            "rcs_m2": 10.0,
        }

        self.decoy_params = {
            "lifetime_s": 15.0,
            "effectiveness": 0.8,
            "decay_rate": 0.05,
            "ir_signature": 500.0,
            "rcs_m2": 5.0,
        }

    def launch_flares(self, simulator=None):
        """
        Launch flares (creates physical countermeasure objects in simulation).

        Args:
            simulator: Simulator instance to add countermeasure to

        Returns:
            Created flare object or None
        """
        if self.flares > 0:
            self.flares -= 1
            flare = create_countermeasure("flare", self.parent, self.flare_params)
            if simulator is not None:
                simulator.add_unit(flare)
                self.active_countermeasures.append(flare)
            self.parent.flare_deployed = True
            logger.debug(f"{self.parent.name} launched flares! Remaining: {self.flares}")
            return flare
        else:
            logger.debug(f"{self.parent.name} has no flares left.")
            return None

    def launch_chaff(self, simulator=None):
        """
        Deploy chaff (creates physical countermeasure objects in simulation).

        Args:
            simulator: Simulator instance to add countermeasure to

        Returns:
            Created chaff object or None
        """
        if self.chaff > 0:
            self.chaff -= 1
            chaff = create_countermeasure("chaff", self.parent, self.chaff_params)
            if simulator is not None:
                simulator.add_unit(chaff)
                self.active_countermeasures.append(chaff)
            self.parent.chaff_deployed = True
            logger.debug(f"{self.parent.name} deployed chaff. Remaining: {self.chaff}")
            return chaff
        else:
            logger.debug(f"{self.parent.name} has no chaff left.")
            return None

    def activate_ecm(self):
        """
        Activate ECM. Unlike flares/chaff, ECM is an aircraft state rather than
        a physical object in the simulation.
        """
        if self.ecm > 0:
            self.parent.ecm_active = True
            logger.debug(f"{self.parent.name} activated ECM. Available: {self.ecm}")
        else:
            logger.debug(f"{self.parent.name} has no ECM available.")

    def deactivate_ecm(self):
        """Deactivate ECM."""
        self.parent.ecm_active = False
        logger.debug(f"{self.parent.name} deactivated ECM.")

    def deploy_decoys(self, simulator=None):
        """
        Deploy decoys (creates physical countermeasure objects in simulation).

        Args:
            simulator: Simulator instance to add countermeasure to

        Returns:
            Created decoy object or None
        """
        if self.decoys > 0:
            self.decoys -= 1
            decoy = create_countermeasure("decoy", self.parent, self.decoy_params)
            if simulator is not None:
                simulator.add_unit(decoy)
                self.active_countermeasures.append(decoy)
            self.parent.decoys_deployed = True
            logger.debug(f"{self.parent.name} deployed decoys. Remaining: {self.decoys}")
            return decoy
        else:
            logger.debug(f"{self.parent.name} has no decoys left.")
            return None

    def cleanup_expired_countermeasures(self):
        """Remove references to countermeasures that have expired."""
        self.active_countermeasures = [
            cm for cm in self.active_countermeasures if not getattr(cm, "should_be_removed", False)
        ]

    def get_nearby_countermeasures(self, position, max_range_m=1000.0):
        """
        Get countermeasures near a given position.

        Args:
            position: Position to check from
            max_range_m: Maximum range to consider

        Returns:
            List of nearby countermeasure objects
        """
        from bvr_marl_core.simulator.utils.geodesics import geodetic_distance_km

        nearby = []
        for cm in self.active_countermeasures:
            if hasattr(cm, "position"):
                distance_m = (
                    geodetic_distance_km(
                        position.lat,
                        position.lon,
                        position.alt,
                        cm.position.lat,
                        cm.position.lon,
                        cm.position.alt,
                    )
                    * 1000.0
                )

                if distance_m <= max_range_m:
                    nearby.append((cm, distance_m))

        nearby.sort(key=lambda x: x[1])
        return nearby
