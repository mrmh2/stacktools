import numpy as np

from imageio import volread

from dtoolbioimage.segment import Segmentation3D


def load_segmentation_from_tif(segmentation_fpath):
    volume = volread(segmentation_fpath)
    transposed = np.transpose(volume, axes=(1, 2, 0))

    segmentation = transposed.view(Segmentation3D)

    return segmentation


def filter_segmentation_by_region_list(segmentation, rids):

    trimmed_segmentation = np.zeros(segmentation.shape, dtype=segmentation.dtype)

    for rid in rids:
        trimmed_segmentation[np.where(segmentation == rid)] = rid

    return trimmed_segmentation.view(Segmentation3D)
