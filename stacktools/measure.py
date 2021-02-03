import logging

import numpy as np
import pandas as pd

from scipy import optimize
from skimage.morphology import ball, erosion, dilation
from skimage.transform import resize

RADIUS=15
ZRADIUS=10
SQUISHBALL = resize(ball(RADIUS).astype(float), (31, 31, 21), order=0)
BALL = ball(RADIUS)


logger = logging.getLogger("stacktools")


def measure_by_sphere_fit(segmentation, measure_stack, rid):
    coords = np.where(segmentation == rid)
    r0, c0, z0 = np.mean(coords, axis=1).astype(int)

    cell_mask = erosion(dilation(segmentation == rid)).astype(int)
    measure_image = cell_mask * measure_stack
    rd, cd, zd = np.where(BALL > 0.5)

    def ball_intensity(params):
        r, c, z = params
        rr, cc, zz = (rd + r - RADIUS).astype(int), (cd + c - RADIUS).astype(int), (zd + z - RADIUS).astype(int)
        total_intensity = int(measure_image[rr, cc, zz].sum())

        return -total_intensity

    r, c, z = optimize.fmin(ball_intensity, (r0, c0, z0), disp=False).astype(int)

    return -ball_intensity((r, c, z)) / len(rd)


def measure_by_obsphere_fit(segmentation, measure_stack, rid):
    coords = np.where(segmentation == rid)
    r0, c0, z0 = np.mean(coords, axis=1).astype(int)

    cell_mask = erosion(dilation(segmentation == rid)).astype(int)
    measure_image = cell_mask * measure_stack
    rd, cd, zd = np.where(SQUISHBALL > 0.5)

    def squishball_intensity(params):
        r, c, z = params
        rr, cc, zz = (rd + r - RADIUS).astype(int), (cd + c - RADIUS).astype(int), (zd + z - 10).astype(int)
        total_intensity = int(measure_image[rr, cc, zz].sum())

        return -total_intensity

    try:
        r, c, z = optimize.fmin(squishball_intensity, (r0, c0, z0), disp=False).astype(int)
    except IndexError:
        return 0, (0, 0, 0)

    mean_in_region = -squishball_intensity((r, c, z)) / len(rd)
    p = r, c, z

    return mean_in_region, p


def measure_by_obsphere_fit_brute(segmentation, measure_stack, rid):
    coords = np.where(segmentation[:-RADIUS,:-RADIUS,:-ZRADIUS] == rid)
    r0, c0, z0 = np.mean(coords, axis=1).astype(int)
    # print(f"Start at {r0, c0, z0}")

    cell_mask = erosion(dilation(segmentation == rid)).astype(int)
    measure_image = cell_mask * measure_stack
    rd, cd, zd = np.where(SQUISHBALL > 0.5)

    def squishball_intensity(params):
        r, c, z = params
        rr, cc, zz = (rd + r - RADIUS).astype(int), (cd + c - RADIUS).astype(int), (zd + z - 10).astype(int)
        total_intensity = int(measure_image[rr, cc, zz].sum())

        return total_intensity

    points = list(zip(*coords))
    all_intensities = [squishball_intensity(p) for p in points]
    max_intensity_index = np.argmax(all_intensities)
    r, c, z = points[max_intensity_index]

    mean_in_region = squishball_intensity((r, c, z)) / len(rd)
    p = r, c, z

    return mean_in_region, p, (r0, c0, z0)


def mask_measure_stack_by_region(segmentation, measure_stack, rid):
    coords = np.where(segmentation[:-RADIUS,:-RADIUS,:-ZRADIUS] == rid)

    r0, c0, z0 = np.mean(coords, axis=1).astype(int)

    cell_mask = erosion(dilation(segmentation == rid)).astype(int)
    measure_image = cell_mask * measure_stack

    return measure_image


def fit_element(measure_image, element_coords, coords):

    rd, cd, zd = element_coords

    def element_intensity(params):
        r, c, z = params
        rr, cc, zz = (rd + r - RADIUS).astype(int), (cd + c - RADIUS).astype(int), (zd + z - 10).astype(int)
        total_intensity = int(measure_image[rr, cc, zz].sum())

        return total_intensity

    points = list(zip(*coords))
    all_intensities = [element_intensity(p) for p in points]
    max_intensity_index = np.argmax(all_intensities)
    r, c, z = points[max_intensity_index]

    return r, c, z


def multi_measure(segmentation, measure_stack, rid):
    
    measure_image = mask_measure_stack_by_region(segmentation, measure_stack, rid)

    n_voxels = np.sum(segmentation == rid)

    element_coords = np.where(SQUISHBALL > 0.5)

    coords = np.where(segmentation[:-RADIUS,:-RADIUS,:-ZRADIUS] == rid)
    cn_seg = tuple(np.mean(coords, axis=1).astype(int))
    p = fit_element(measure_image, element_coords, coords)

    rd, cd, zd = element_coords

    def element_coords_at_point(p):
        r, c, z = p
        rr, cc, zz = (rd + r - RADIUS).astype(int), (cd + c - RADIUS).astype(int), (zd + z - 10).astype(int)
        return rr, cc, zz
    
    rr, cc, zz = element_coords_at_point(p)

    points_in_element = set(zip(rr, cc, zz))
    points_in_region = set(zip(*coords))

    points_outside_element = points_in_region - points_in_element
    if len(points_outside_element):
        re, ce, ze = zip(*points_outside_element)
        mean_outside_element = measure_image[re, ce, ze].mean()
    else:
        mean_outside_element = None

    overlap_points = points_in_element & points_in_region
    overlap_fraction = len(overlap_points) / len(points_in_element)
    mean_in_element = measure_image[rr, cc, zz].mean()

    measures = {
        'cell_voxels': n_voxels,
        'sphere_fit_centroid': p,
        'segmented_cell_centroid': cn_seg,
        'mean_in_sphere': mean_in_element,
        'mean_outside_sphere': mean_outside_element,
        'overlap_fraction': overlap_fraction
    }

    return measures


def measure_with_sphere_fitting(sms, rid):

    measure_image = mask_measure_stack_by_region(
        sms.segmentation, sms.measure_stack, rid)

    element_coords = np.where(SQUISHBALL > 0.5)

    coords = np.where(sms.segmentation[:-RADIUS, :-RADIUS, :-ZRADIUS] == rid)
    cn_seg = tuple(np.mean(coords, axis=1).astype(int))

    if len(coords[0]) == 0:
        logging.info(f"Failed to measure {rid}")
        return None

    p = fit_element(measure_image, element_coords, coords)

    rd, cd, zd = element_coords

    def element_coords_at_point(p):
        r, c, z = p
        rr, cc, zz = (rd + r - RADIUS).astype(int), (cd + c -
                                                     RADIUS).astype(int), (zd + z - 10).astype(int)
        return rr, cc, zz

    rr, cc, zz = element_coords_at_point(p)

    points_in_element = set(zip(rr, cc, zz))
    points_in_region = set(zip(*coords))

    points_outside_element = points_in_region - points_in_element
    if len(points_outside_element):
        re, ce, ze = zip(*points_outside_element)
        mean_outside_element = measure_image[re, ce, ze].mean()
    else:
        mean_outside_element = None

    overlap_points = points_in_element & points_in_region
    overlap_fraction = len(overlap_points) / len(points_in_element)
    mean_in_element = measure_image[rr, cc, zz].mean()

    measures = {
        'sphere_fit_centroid': p,
        'mean_in_sphere': mean_in_element,
        'mean_outside_sphere': mean_outside_element,
        'overlap_fraction': overlap_fraction
    }

    return measures


def measure_rid(sms, rid):
    coords = np.where(sms.segmentation == rid)
    return np.mean(sms.measure_stack[coords])


def measure_properties(sms, rid):
    mean_intensity = measure_rid(sms, rid)

    measurement = {
        "label": rid,
        "mean_intensity": mean_intensity
    }

    return measurement


def measure_region_by_rid(sms, rid):
    coords = np.where(sms.segmentation == rid)
    return np.mean(sms.measure_stack[coords])


def measure_plane_mean(sms, rid):
    rcoords = np.where(sms.segmentation == rid)
    cn_r, cn_c, cn_z = [int(a.mean()) for a in rcoords]

    measure_plane_mean = sms.measure_stack[:, :, cn_z].mean()

    return {
        "z_mean": measure_plane_mean
    }


def measure_basic_properties(sms, rid):

    mean_intensity = measure_region_by_rid(sms, rid)
    total_volume = (sms.segmentation == rid).sum()

    measurement = {
        "label": rid,
        "mean_intensity": mean_intensity,
        "total_volume": total_volume,
    }

    return measurement


def measure_centroid(sms, rid):

    rcoords = np.where(sms.segmentation == rid)
    cn_r, cn_c, cn_z = [int(a.mean()) for a in rcoords]

    return {
        "cn_r": cn_r,
        "cn_c": cn_c,
        "cn_z": cn_z
    }


def measure_many_properties(sms, rid):

    measure_funcs = [
        measure_basic_properties,
        measure_centroid,
        measure_plane_mean,
        measure_with_sphere_fitting
    ]

    measures = {}

    for measure_func in measure_funcs:
        new_measures = measure_func(sms, rid)
        if new_measures is not None:
            measures.update(new_measures)
        else:
            return None

    return measures


def measure_all_regions(sms):
    """Measure all of the regions in a segmentation measure stack."""

    measures = []
    for rid in sms.segmentation.labels:
        logger.info(f"Measuring {rid}")
        properties = measure_many_properties(sms, rid)
        if properties is not None:
            properties['label'] = sms.label_lookup[rid]
            properties['region_id'] = rid
            measures.append(properties)

    return pd.DataFrame(measures)




