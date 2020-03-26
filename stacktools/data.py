import numpy as np  
from skimage.morphology import dilation

from dtoolbioimage import ImageDataSet, zoom_to_match_scales, scale_to_uint8

from stacktools.cache import fn_caching_wrapper


def get_masked_venus_stack(image_ds_uri, root_name):
    venus_stack = get_stack_by_name(image_ds_uri, root_name)
    wall_stack = get_stack_by_name(image_ds_uri, root_name, channel=1)

    base_mask = dilation(scale_to_uint8(wall_stack) > 100)
    venus_stack[np.where(base_mask)] = 0

    return venus_stack


@fn_caching_wrapper
def get_stack_by_name(ids_uri, root_name, channel=0):

    ids = ImageDataSet.from_uri(ids_uri)
    name_lookup = dict(ids.get_image_series_name_pairs())
    series_name = name_lookup[root_name]

    raw_stack = ids.get_stack(root_name, series_name, channel=channel)
    zoomed_stack = zoom_to_match_scales(raw_stack)

    return zoomed_stack
