"""Numba/numpy box ops (corners, collision test, points-in-rbbox).
Used by ObjectPaste (DataBaseSampler) and tools/create_gt_database.py.
"""

import warnings

import numba
import numpy as np

try:
    from numba import errors as _numba_errors
    warnings.filterwarnings("ignore", category=_numba_errors.NumbaPerformanceWarning)
except Exception:
    pass

__all__ = [
    "points_in_rbbox",
    "center_to_corner_box2d",
    "center_to_corner_box3d",
    "corner_to_surfaces_3d",
    "corner_to_surfaces_3d_jit",
    "points_in_convex_polygon_3d_jit",
    "corner_to_standup_nd_jit",
    "box2d_to_corner_jit",
]


# ---------------------------------------------------------------------------
# Non-jit helper functions
# ---------------------------------------------------------------------------


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point."""
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def rotation_2d(points, angles):
    """Rotation 2d points based on origin point clockwise when angle positive."""
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_3d_in_axis(points, angles, axis=0):
    """Rotate points in specific axis."""
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [[rot_cos, zeros, -rot_sin], [zeros, ones, zeros], [rot_sin, zeros, rot_cos]]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [[rot_cos, -rot_sin, zeros], [rot_sin, rot_cos, zeros], [zeros, zeros, ones]]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [[zeros, rot_cos, -rot_sin], [zeros, rot_sin, rot_cos], [ones, zeros, zeros]]
        )
    else:
        raise ValueError("axis should in range")
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations with shape (N, 2).
        dims (np.ndarray): Dimensions with shape (N, 2).
        angles (np.ndarray, optional): Rotation with shape (N).
        origin (float): Origin point. Defaults to 0.5.

    Returns:
        np.ndarray: Corners with shape (N, 4, 2).
    """
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 1.0, 0.5), axis=1):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations with shape (N, 3).
        dims (np.ndarray): Dimensions with shape (N, 3).
        angles (np.ndarray, optional): Rotation with shape (N).
        origin (tuple): Origin point. Defaults to (0.5, 1.0, 0.5).
        axis (int): Rotation axis. Defaults to 1.

    Returns:
        np.ndarray: Corners with shape (N, 8, 3).
    """
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def corner_to_surfaces_3d(corners):
    """Convert 3d box corners to surfaces that normal vectors all direct
    to internal.

    Args:
        corners (np.ndarray): 3d box corners with shape (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with shape (N, 6, 4, 3).
    """
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces


# ---------------------------------------------------------------------------
# JIT-compiled functions
# ---------------------------------------------------------------------------


@numba.jit(nopython=True)
def corner_to_standup_nd_jit(boxes_corner):
    """Convert boxes_corner to aligned (min-max) boxes.

    Args:
        boxes_corner (np.ndarray): Boxes corners with shape (N, 2**dim, dim).

    Returns:
        np.ndarray: Aligned (min-max) boxes with shape (N, dim*2).
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    """Convert box2d to corner.

    Args:
        boxes (np.ndarray): Boxes2d with rotation, shape (N, 5).

    Returns:
        np.ndarray: Box corners with shape (N, 4, 2).
    """
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """Convert 3d box corners to surfaces (jit version).

    Args:
        corners (np.ndarray): 3d box corners with shape (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with shape (N, 6, 4, 3).
    """
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array(
        [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7]
    ).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def surface_equ_3d(polygon_surfaces):
    """Compute surface equations for 3d polygons.

    Args:
        polygon_surfaces (np.ndarray): shape [num_polygon, max_surfaces, 4, 3]

    Returns:
        tuple: (normal_vec, d) where normal_vec has shape
            [num_polygon, max_surfaces, 3] and d has shape
            [num_polygon, max_surfaces].
    """
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    d = np.einsum("aij, aij->ai", normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


@numba.njit
def points_in_convex_polygon_3d_kernel(
    points, polygon_surfaces, normal_vec, d, num_surfaces
):
    """Check if points are inside convex polygons (jit inner function)."""
    max_num_surfaces = polygon_surfaces.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """Check points in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape
            (num_polygon, max_surfaces, max_points_of_surface, 3).
        num_surfaces (np.ndarray, optional): Number of surfaces per polygon.

    Returns:
        np.ndarray: Result matrix with shape (num_points, num_polygon).
    """
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    return points_in_convex_polygon_3d_kernel(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0)):
    """Check points in rotated bbox and return boolean mask.

    Uses direct rotation-based approach (no surface normals) —
    works correctly for any origin convention.

    Args:
        points (np.ndarray): Points with shape (N, 3+dim).
        rbbox (np.ndarray): Boxes3d with rotation, shape (M, 7+).
            Columns: x, y, z, x_size, y_size, z_size, yaw, ...
        z_axis (int): Which axis is height. Defaults to 2.
        origin (tuple): Position of box center. Defaults to (0.5, 0.5, 0).

    Returns:
        np.ndarray: Boolean matrix with shape (N, M), True = inside.
    """
    # Translate points relative to each box center
    # points: (N, 3+), rbbox: (M, 7+)
    N = points.shape[0]
    M = rbbox.shape[0]
    if M == 0:
        return np.zeros((N, 0), dtype=bool)

    # Box centers
    centers = rbbox[:, :3]  # (M, 3)
    dims = rbbox[:, 3:6]   # (M, 3) = (x_size, y_size, z_size)
    yaw = rbbox[:, 6]      # (M,)

    # For origin (0.5, 0.5, 0), the center IS the reference point.
    # For other origins, adjust the center position.
    origin = np.array(origin, dtype=points.dtype)
    ref_offset = dims * (np.array([0.5, 0.5, 0]) - origin)
    centers = centers + ref_offset

    # Compute relative positions: (N, M, 3)
    diff = points[:, np.newaxis, :3] - centers[np.newaxis, :, :]

    # Rotate points into box-local frame
    # The box corners are generated by applying R^T to unit corners.
    # So to inverse-transform, we apply R (non-transposed).
    cos_yaw = np.cos(yaw)  # (M,)
    sin_yaw = np.sin(yaw)  # (M,)

    # R @ diff (inverse of R^T rotation)
    local_x = diff[:, :, 0] * cos_yaw + diff[:, :, 1] * sin_yaw
    local_y = -diff[:, :, 0] * sin_yaw + diff[:, :, 1] * cos_yaw
    local_z = diff[:, :, 2]

    # Check axis-aligned containment
    half_x = dims[:, 0] / 2.0  # (M,)
    half_y = dims[:, 1] / 2.0
    half_z = dims[:, 2] / 2.0

    inside = (
        (np.abs(local_x) <= half_x)
        & (np.abs(local_y) <= half_y)
        & (np.abs(local_z) <= half_z)
    )
    return inside


# ---------------------------------------------------------------------------
# Additional functions needed by noise_per_object_v3_
# ---------------------------------------------------------------------------


@numba.njit
def rotation_box2d_kernel(corners, angle, rot_mat_T):
    """Rotate 2D boxes."""
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.

    Args:
        boxes (np.ndarray): Corners of current boxes, shape (N, 4, 2).
        qboxes (np.ndarray): Boxes to be avoid colliding, shape (K, 4, 2).
        clockwise (bool): Whether corners are clockwise. Default: True.

    Returns:
        np.ndarray: Collision matrix of shape (N, K).
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]), axis=2)
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]
                )
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True
                                    break
                        if ret[i, j]:
                            break
                    if not ret[i, j]:
                        box_overlap_qbox = True
                        for box_l in range(4):
                            for k in range(4):
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if not box_overlap_qbox:
                                break
                        if not box_overlap_qbox:
                            qbox_overlap_box = True
                            for box_l in range(4):
                                for k in range(4):
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:
                                        qbox_overlap_box = False
                                        break
                                if not qbox_overlap_box:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True
                        else:
                            ret[i, j] = True
    return ret


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    """Add noise to every box (only on the horizontal plane)."""
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                rotation_box2d_kernel(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners
                )
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises, global_rot_noises):
    """Add noise to every box with global rotations."""
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += dst_grot - current_grot
                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = (
                    current_box[0, 2:4] * corners_norm @ rot_mat_T
                    + current_box[0, :2]
                )
                current_corners -= current_box[0, :2]
                rotation_box2d_kernel(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners
                )
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += dst_pos - boxes[i, :2]
                    rot_noises[i, j] += dst_grot - current_grot
                    break
    return success_mask


def select_transform(transform, indices):
    """Select transform based on indices."""
    result = np.zeros(
        (transform.shape[0], *transform.shape[2:]), dtype=transform.dtype
    )
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@numba.njit
def rotation_matrix_3d(rot_mat_T, angle, axis):
    """Get the 3D rotation matrix."""
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos


@numba.njit
def points_transform_(
    points, centers, point_masks, loc_transform, rot_transform, valid_mask
):
    """Apply transforms to points and box centers."""
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        rotation_matrix_3d(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i : i + 1, :3] = points[i : i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    """Transform 3D boxes."""
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def noise_per_object_v3_(
    gt_boxes,
    points=None,
    valid_mask=None,
    rotation_perturb=np.pi / 4,
    center_noise_std=1.0,
    global_random_rot_range=np.pi / 4,
    num_try=100,
):
    """Random rotate or remove each groundtruth independently.

    Args:
        gt_boxes (np.ndarray): Ground truth boxes with shape (N, 7+).
        points (np.ndarray, optional): Input point cloud with shape (M, 4).
        valid_mask (np.ndarray, optional): Mask for valid boxes.
        rotation_perturb (float): Rotation perturbation.
        center_noise_std (float): Center noise standard deviation.
        global_random_rot_range (float): Global random rotation range.
        num_try (int): Number of tries.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range, global_random_rot_range]
    enable_grot = (
        np.abs(global_random_rot_range[0] - global_random_rot_range[1]) >= 1e-3
    )

    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)

    loc_noises = np.random.normal(
        scale=center_noise_std, size=[num_boxes, num_try, 3]
    )
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try]
    )
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try],
    )

    origin = (0.5, 0.5, 0)
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2
    )

    if not enable_grot:
        selected_noise = noise_per_box(
            gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises
        )
    else:
        selected_noise = noise_per_box_v2_(
            gt_boxes[:, [0, 1, 3, 4, 6]],
            valid_mask,
            loc_noises,
            rot_noises,
            global_rot_noises,
        )

    loc_transforms = select_transform(loc_noises, selected_noise)
    rot_transforms = select_transform(rot_noises, selected_noise)
    surfaces = corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(
            points,
            gt_boxes[:, :3],
            point_masks,
            loc_transforms,
            rot_transforms,
            valid_mask,
        )

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)
