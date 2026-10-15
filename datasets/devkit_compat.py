"""shapely 2.x compatibility shims for nuscenes-devkit 1.2 map_expansion.

shapely 2 removed iteration over Multi* geometries and geometry truthiness;
devkit's NuScenesMapExplorer.mask_for_lines still does `for line in lines`
on MultiLineString and mask_for_polygons does `if not polygons`.

Import this module (apply_compat()) before rasterizing map masks.
"""

import numpy as np

PATCHED = False


def apply_compat():
    global PATCHED
    if PATCHED:
        return
    try:
        import cv2
        from shapely.geometry import LineString, MultiPolygon
        from nuscenes.map_expansion.map_api import NuScenesMapExplorer
    except ImportError:
        return

    @staticmethod
    def mask_for_lines(lines: LineString, mask: np.ndarray) -> np.ndarray:
        """Convert a Shapely LineString/MultiLineString to an image mask
        (shapely 2.x safe)."""
        if lines.is_empty:
            return mask
        if lines.geom_type == "MultiLineString":
            geoms = lines.geoms  # shapely 2: explicit .geoms
        else:
            geoms = [lines]
        for line in geoms:
            coords = np.asarray(list(line.coords), np.int32).reshape((-1, 2))
            cv2.polylines(mask, [coords], False, 1, 2)
        return mask

    @staticmethod
    def mask_for_polygons(polygons: MultiPolygon, mask: np.ndarray) -> np.ndarray:
        """Convert a Polygon/MultiPolygon to an image mask (shapely 2 safe:
        no geometry truthiness)."""
        if polygons is None or polygons.is_empty:
            return mask

        def int_coords(x):
            return np.array(x).round().astype(np.int32)

        if polygons.geom_type == "MultiPolygon":
            geoms = polygons.geoms
        else:
            geoms = [polygons]
        exteriors = [int_coords(poly.exterior.coords) for poly in geoms]
        interiors = [int_coords(pi.coords) for poly in geoms for pi in poly.interiors]
        cv2.fillPoly(mask, exteriors, 1)
        cv2.fillPoly(mask, interiors, 0)
        return mask

    NuScenesMapExplorer.mask_for_lines = mask_for_lines
    NuScenesMapExplorer.mask_for_polygons = mask_for_polygons
    PATCHED = True
