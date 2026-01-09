from geographiclib.geodesic import Geodesic
import numpy as np

class MapLimits:
    def __init__(self, left_lon, bottom_lat, right_lon, top_lat, min_alt=0, max_alt=10000):
        self.left_lon = left_lon
        self.bottom_lat = bottom_lat
        self.right_lon = right_lon
        self.top_lat = top_lat
        self.min_alt = min_alt    # Minimum altitude (meters)
        self.max_alt = max_alt    # Maximum altitude (meters)

    def latitude_extent(self):
        return self.top_lat - self.bottom_lat

    def longitude_extent(self):
        return self.right_lon - self.left_lon

    def altitude_extent(self):
        return self.max_alt - self.min_alt

    def max_latitude_extent_km(self):
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.top_lat, self.left_lon,
                                    outmask=Geodesic.DISTANCE)
        d2 = Geodesic.WGS84.Inverse(self.bottom_lat, self.right_lon, self.top_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        return max(d1["s12"] / 1000.0, d2["s12"] / 1000.0)

    def max_longitude_extent_km(self):
        d1 = Geodesic.WGS84.Inverse(self.bottom_lat, self.left_lon, self.bottom_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        d2 = Geodesic.WGS84.Inverse(self.top_lat, self.left_lon, self.top_lat, self.right_lon,
                                    outmask=Geodesic.DISTANCE)
        return max(d1["s12"] / 1000.0, d2["s12"] / 1000.0)

    def relative_position(self, lat, lon, alt):
        lat_rel = (lat - self.bottom_lat) / self.latitude_extent()
        lon_rel = (lon - self.left_lon) / self.longitude_extent()
        alt_rel = (alt - self.min_alt) / self.altitude_extent()
        return np.clip(lat_rel, 0, 1), np.clip(lon_rel, 0, 1), np.clip(alt_rel, 0, 1)

    def absolute_position(self, lat_rel, lon_rel, alt_rel):
        lat = lat_rel * self.latitude_extent() + self.bottom_lat
        lon = lon_rel * self.longitude_extent() + self.left_lon
        alt = alt_rel * self.altitude_extent() + self.min_alt
        return lat, lon, alt

    def in_boundary(self, lat, lon, alt):
        return bool(self.left_lon <= lon <= self.right_lon and
                    self.bottom_lat <= lat <= self.top_lat and
                    self.min_alt <= alt <= self.max_alt)
