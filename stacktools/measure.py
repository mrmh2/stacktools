import numpy as np

from scipy import optimize
from skimage.morphology import ball, erosion, dilation
from skimage.transform import resize

RADIUS=15
ZRADIUS=10
SQUISHBALL = resize(ball(RADIUS).astype(float), (31, 31, 21), order=0)
BALL = ball(RADIUS)


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
