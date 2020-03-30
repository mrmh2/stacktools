import os

import numpy as np

from skimage import img_as_float
from skimage.morphology import erosion, dilation, reconstruction
from skimage.restoration import denoise_tv_chambolle

from dtoolbioimage import Image, Image3D, scale_to_uint8

from scipy.ndimage import gaussian_filter

from imageio import volread

from stacktools.cache import fn_caching_wrapper

def scale_factors(array):

    scaled = array.astype(np.float32)

    return scaled.min(), scaled.max() - scaled.min()


def centroid_project(stack, segmentation, centroids_by_rid):
    canvas = np.zeros((512, 1024), dtype=np.float32)

    coords_vals = []
    for rid, centroid in centroids_by_rid.items():
        r, c, z = centroid
        closed_region = erosion(dilation(segmentation[:,:,int(z)] == rid))
        rr, cc = np.where(closed_region)
        coords_vals.append(((rr, cc), stack[rr, cc, int(z)]))

    smin, srange = scale_factors(np.concatenate([v for c, v in coords_vals]))
    for (rr, cc), val in coords_vals:
        canvas[rr, cc] = ((val - smin) / srange)

    return canvas.view(Image)


def trim_segmentation(segmentation, files):
    trimmed_segmentation = segmentation.copy()

    rids_in_files = set(sum(files.values(), []))

    rids_not_in_files = segmentation.labels - rids_in_files

    for rid in rids_not_in_files:
        trimmed_segmentation[np.where(trimmed_segmentation == rid)] = 0

    return trimmed_segmentation


@fn_caching_wrapper
def filter_segmentation_by_regions(segmentation, region_ids):

    rids_not_in_files = segmentation.labels - set(region_ids)

    trimmed_segmentation = segmentation.copy()

    for rid in rids_not_in_files:
        trimmed_segmentation[np.where(trimmed_segmentation == rid)] = 0

    return trimmed_segmentation


@fn_caching_wrapper
def calculate_hdome(input_stack):
    stack_float = img_as_float(input_stack)
    stack_gauss = gaussian_filter(stack_float, 3)

    h = 0.015
    seed = stack_gauss - h

    mask = stack_gauss
    dilated = reconstruction(seed, mask, method='dilation')

    hdome2 = stack_gauss - 0.5 * dilated

    return hdome2.astype(np.float32).view(Image3D)


@fn_caching_wrapper
def denoise_tv_chambolle_f32(stack, weight=0.01):
    return denoise_tv_chambolle(stack, weight=weight).astype(np.float32)




def get_wall_mask(image_ds_uri, root_name):
    wall_stack = get_stack_by_name(image_ds_uri, root_name, channel=1)

    mask = erosion(scale_to_uint8(wall_stack) < 100)

    return mask
