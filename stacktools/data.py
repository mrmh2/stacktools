from dtoolbioimage import ImageDataSet, zoom_to_match_scales

from stacktools.utils import fn_caching_wrapper


@fn_caching_wrapper
def get_stack_by_name(ids_uri, root_name, channel=0):

    ids = ImageDataSet.from_uri(ids_uri)
    name_lookup = dict(ids.get_image_series_name_pairs())
    series_name = name_lookup[root_name]

    raw_stack = ids.get_stack(root_name, series_name, channel=channel)
    zoomed_stack = zoom_to_match_scales(raw_stack)

    return zoomed_stack
