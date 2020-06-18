import numpy as np

from imageio import volread

from dtoolbioimage.segment import Segmentation3D

from stacktools.cache import fn_caching_wrapper


def load_segmentation_from_tif(segmentation_fpath):
    volume = volread(segmentation_fpath)
    transposed = np.transpose(volume, axes=(1, 2, 0))

    segmentation = transposed.view(Segmentation3D)

    return segmentation


@fn_caching_wrapper
def filter_segmentation_by_region_list(segmentation, region_ids):

    rids_not_in_files = segmentation.labels - set(region_ids)

    trimmed_segmentation = segmentation.copy()

    for rid in rids_not_in_files:
        trimmed_segmentation[np.where(trimmed_segmentation == rid)] = 0

    return trimmed_segmentation