import numpy as np

from scipy import optimize
from skimage.morphology import ball, erosion, dilation
from skimage.transform import resize


# ratio = float(venus.metadata.PhysicalSizeX) / float(venus.metadata.PhysicalSizeZ)
# ball(15).shape[0] * ratio
RADIUS=15
SQUISHBALL = resize(ball(15).astype(float), (31, 31, 21), order=0)
BALL = ball(15)


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
