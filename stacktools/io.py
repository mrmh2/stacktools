from dtoolbioimage import ImageDataSet, zoom_to_match_scales
from dtoolbioimage.segment import Segmentation3D


from stacktools.cache import fn_caching_wrapper


@fn_caching_wrapper
def get_stack_by_imname_sname(ids_uri, imname, sname, channel=0):

    ids = ImageDataSet.from_uri(ids_uri)

    raw_stack = ids.get_stack(imname, sname, channel=channel)
    zoomed_stack = zoom_to_match_scales(raw_stack)

    return zoomed_stack


@fn_caching_wrapper
def load_segmentation3d_from_file(fpath):

    segmentation = Segmentation3D.from_file(fpath)

    return segmentation