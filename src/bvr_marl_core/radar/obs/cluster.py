import math
from collections.abc import Mapping


class Clusterer:
    """Clusters raw radar detections into bins based on range and angles."""

    def __init__(self, angular_res_deg: float, range_res_m: float):
        self.angular_res_deg = angular_res_deg
        self.range_res_m = range_res_m

    def cluster(self, dets: list[Mapping] | tuple[Mapping, ...]) -> list[dict]:
        """Group detections whose (azimuth, elevation, range) fall into the same discrete bins."""
        if not dets:
            return []

        # Detection counts per tick are tiny (a handful per radar), so plain Python
        # aggregation is much faster than allocating numpy arrays per call.
        inv_ang = 1.0 / self.angular_res_deg
        inv_rng = 1.0 / self.range_res_m

        bin_dict: dict[tuple[int, int, int], list[Mapping]] = {}
        for d in dets:
            az = float(d["az"])
            el = float(d["el"])
            di = float(d["d"])
            if not (math.isfinite(az) and math.isfinite(el) and math.isfinite(di)):
                continue
            key = (
                math.floor(az * inv_ang),
                math.floor(el * inv_ang),
                math.floor(di * inv_rng),
            )
            members = bin_dict.get(key)
            if members is None:
                bin_dict[key] = [d]
            else:
                members.append(d)

        clusters = []
        for members in bin_dict.values():
            m = len(members)
            az_sum = el_sum = d_sum = dop_sum = lat_sum = lon_sum = alt_sum = 0.0
            is_deception = False
            engagement_ids = []
            jammer_ids = []
            t_list = []
            report_ids = []
            source_ids = []
            report_lineage = []
            acquisition_times = []
            for d in members:
                az_sum += float(d["az"])
                el_sum += float(d["el"])
                d_sum += float(d["d"])
                dop_sum += float(d.get("dop", 0.0))
                lat_sum += float(d.get("lat", 0.0))
                lon_sum += float(d.get("lon", 0.0))
                alt_sum += float(d.get("alt", 0.0))
                if d.get("is_deception", False):
                    is_deception = True
                eid = d.get("engagement_id")
                if eid is not None:
                    engagement_ids.append(eid)
                jid = d.get("jammer_id")
                if jid is not None:
                    jammer_ids.append(jid)
                t_list.append(d.get("T", None))
                if d.get("report_id") is not None:
                    report_ids.append(int(d["report_id"]))
                if d.get("source_id") is not None:
                    source_ids.append(d["source_id"])
                lineage = d.get("report_lineage")
                if lineage:
                    report_lineage.extend(lineage)
                elif d.get("source_id") is not None and d.get("report_id") is not None:
                    report_lineage.append((d["source_id"], int(d["report_id"])))
                if d.get("acquisition_time_s") is not None:
                    acquisition_times.append(float(d["acquisition_time_s"]))

            engagement_id = (
                max(set(engagement_ids), key=engagement_ids.count) if engagement_ids else None
            )
            jammer_id = max(set(jammer_ids), key=jammer_ids.count) if jammer_ids else None

            cluster = {
                "az": az_sum / m,
                "el": el_sum / m,
                "d": d_sum / m,
                "dop": dop_sum / m,
                "lat": lat_sum / m,
                "lon": lon_sum / m,
                "alt": alt_sum / m,
                "T": t_list,
                "is_deception": is_deception,
                "engagement_id": engagement_id,
                "jammer_id": jammer_id,
                "report_ids": tuple(report_ids),
                "source_ids": tuple(sorted(set(source_ids), key=str)),
                "report_lineage": tuple(
                    sorted(set(report_lineage), key=lambda item: (str(item[0]), item[1]))
                ),
                "acquisition_time_s": max(acquisition_times) if acquisition_times else None,
            }

            # Never set 'T' on deception clusters (enforce no trusted data on ghosts)
            if is_deception:
                cluster["T"] = None

            clusters.append(cluster)
        return clusters
