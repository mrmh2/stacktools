import os
import ast
import logging
import pathlib
import dataclasses

import dtoolcore
import imageio
import numpy as np  
import pandas as pd


from dbimage.io import read_dbim_from_fpath


from skimage.morphology import dilation

from dtoolbioimage import ImageDataSet, zoom_to_match_scales, scale_to_uint8, Image3D
from dtoolbioimage.segment import Segmentation3D

from stacktools.cache import fn_caching_wrapper
from stacktools.utils import (
    denoise_tv_chambolle_f32,
    calculate_hdome,
    filter_segmentation_by_regions
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SegmentationMeasureStack:

    segmentation: Segmentation3D
    measure_stack: Image3D
    wall_stack: Image3D


class InitialsSMS(SegmentationMeasureStack):

    @classmethod
    def from_config_and_spec(cls, config, spec):
        wall_stack = get_zoomed_stack_cached(
            config.ids_uri,
            spec.image_name,
            spec.series_name,
            config.wall_channel
        )
        measure_stack = get_zoomed_stack_cached(
            config.ids_uri,
            spec.image_name,
            spec.series_name,
            config.measure_channel
        )
        segmentation, label_lookup = load_segmentation_and_label_lookup(
            config, spec)

        sms = cls(segmentation, measure_stack, wall_stack)
        sms.label_lookup = label_lookup

        return sms


def load_segmentation_and_label_lookup(config, spec):

    base_dirpath = pathlib.Path(config.segmentation_dirpath)
    region_id_fpath = base_dirpath/spec.fname

    segmentation_fpath = str(region_id_fpath).rsplit(
        '.', maxsplit=1)[0] + '.tif'
    logger.info(f"Loading segmentation from {segmentation_fpath}")
    s3d = get_segmentation(segmentation_fpath).view(Segmentation3D)

    df = pd.read_csv(region_id_fpath, names=["rid", "label"])
    valid_regions = set(df[df.label != 0].rid)

    label_lookup = df[df.label != 0].set_index("rid").label.to_dict()

    trimmed_segmentation = filter_segmentation_by_region_list(
        s3d, valid_regions).view(Segmentation3D)

    return trimmed_segmentation, label_lookup


@fn_caching_wrapper
def get_zoomed_stack_cached(ids_uri, image_name, series_name, channel):
    stack = get_stack_cached(ids_uri, image_name, series_name, channel)
    return zoom_to_match_scales(stack)


@fn_caching_wrapper
def get_stack_cached(ids_uri, image_name, series_name, channel):
    image_ds = ImageDataSet(ids_uri)

    return image_ds.get_stack(
        image_name,
        series_name,
        0,
        channel=channel
    )


@fn_caching_wrapper
def filter_segmentation_by_region_list(segmentation, region_ids):

    rids_not_in_files = segmentation.labels - set(region_ids)

    trimmed_segmentation = segmentation.copy()

    for rid in rids_not_in_files:
        trimmed_segmentation[np.where(trimmed_segmentation == rid)] = 0

    return trimmed_segmentation


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


@fn_caching_wrapper
def get_segmentation(segmentation_fpath):

    volume = imageio.volread(segmentation_fpath)
    transposed = np.transpose(volume, axes=(1, 2, 0))

    return transposed


class FCADataLoader(object):

    def __init__(self, image_ds_uri, root_data_dirpath, seg_data_dirpath):
        self.image_ds_uri = image_ds_uri
        self.root_data_dirpath = root_data_dirpath
        self.seg_data_dirpath = seg_data_dirpath

    def load_root(self, root_name):
        venus_stack = get_stack_by_name(self.image_ds_uri, root_name)
        wall_stack = get_stack_by_name(self.image_ds_uri, root_name, channel=1)
        segmentation = get_segmentation(self.seg_data_dirpath, root_name)
        root_data_fpath = os.path.join(
            self.root_data_dirpath, f"{root_name}-spherefit.csv"
            )
        root_data = pd.read_csv(root_data_fpath)

        return FCARootData(root_name, wall_stack, venus_stack, segmentation, root_data)


class FCADataSetLoader(object):

    def __init__(self, image_ds_uri, seg_ds_uri, root_data_dirpath):
        self.image_ds_uri = image_ds_uri
        self.seg_ds_uri = seg_ds_uri
        self.root_data_dirpath = root_data_dirpath

    def load_root(self, root_name):
        venus_stack = get_stack_by_name(self.image_ds_uri, root_name)
        wall_stack = get_stack_by_name(self.image_ds_uri, root_name, channel=1)

        ds = dtoolcore.DataSet.from_uri(self.seg_ds_uri)
        idn = dtoolcore.utils.generate_identifier(f"{root_name}.dbim")
        fpath = ds.item_content_abspath(idn)
        segmentation = read_dbim_from_fpath(fpath)

        root_data_fpath = os.path.join(
            self.root_data_dirpath, f"{root_name}-spherefit.csv"
        )
        root_data = pd.read_csv(root_data_fpath)

        return FCARootData(root_name, wall_stack, venus_stack, segmentation, root_data)


class FCARootData(object):

    def __init__(self, name, wall_stack, venus_stack, segmentation, root_data):
        self.wall_stack = wall_stack
        self.venus_stack = venus_stack
        self.segmentation = segmentation.view(Segmentation3D)
        self.root_data = root_data
        self.name = name

    @property
    def denoised_venus_stack(self):
        return denoise_tv_chambolle_f32(self.venus_stack)

    @property
    def hdome_venus_stack(self):
        return calculate_hdome(self.denoised_venus_stack)

    @property
    def files(self):
        return set(self.root_data.file_id)

    def regions_in_file(self, fid):
        return self.cell_centroids(fid).keys()

    # TODO - should probably be regions_in_any_file
    @property
    def regions_in_files(self):
        return set(self.root_data.region_id)

    def cell_centroids(self, fid):
        centroid_by_rid_raw = self.root_data[self.root_data.file_id == fid].set_index('region_id').cell_centroid.to_dict()
        centroid_by_rid = {rid: ast.literal_eval(cn) for rid, cn in centroid_by_rid_raw.items()}

        return centroid_by_rid

    def sphere_centroids(self, fid):
        centroid_by_rid_raw = self.root_data[self.root_data.file_id == fid].set_index('region_id').sphere_fit_centroid.to_dict()
        centroid_by_rid = {rid: ast.literal_eval(cn) for rid, cn in centroid_by_rid_raw.items()}

        return centroid_by_rid

    @property
    def trimmed_segmentation(self):
        return filter_segmentation_by_regions(self.segmentation, self.regions_in_files)

    
    def file_coords(self, fid):
        rids = self.regions_in_file(fid)
        coord_blocks = [np.where(self.trimmed_segmentation == rid) for rid in rids]

        rs, cs, zs = list(zip(*coord_blocks))

        rr = np.concatenate(rs)
        cc = np.concatenate(cs)
        zz = np.concatenate(zs)

        return rr, cc, zz
