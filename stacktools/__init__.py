import os
import ast
import logging

from scipy.ndimage.morphology import binary_dilation, binary_erosion
import pandas as pd
from collections import defaultdict

import numpy as np
from imageio import volread

from dtoolbioimage import Image, ImageDataSet

from dtoolbioimage import zoom_to_match_scales

from dtoolbioimage.segment import Segmentation3D


class DataLoader(object):

    def __init__(self, dataset_uri, segmentations_dirpath):
        self.ids = ImageDataSet(dataset_uri)
        self.segmentations_dirpath = segmentations_dirpath

        self.name_lookup = dict(self.ids.get_image_series_name_pairs())


    @property
    def root_names(self):

        return list(self.name_lookup.keys())

    def get_stack_by_series_name(self, series_name):

        series_name = series_name
        image_name = self.name_lookup[series_name]
        
        return self.ids.get_stack(series_name, image_name)

    def load_file_data(self, file_fpath):

        df = pd.read_csv(file_fpath, names=['rid', 'file'])
        file_lookup = pd.Series(df.file.values,index=df.rid).to_dict()
        fids_to_rids = defaultdict(list)
        for rid, fid in file_lookup.items():
            if fid != 0:
                fids_to_rids[fid].append(rid)
        
        return fids_to_rids

    def load_by_name(self, name):
        base_dirpath = os.path.join(self.segmentations_dirpath, name)

        file_info_fname = "Root_segments.tif.csv"
        segmentation_name = "Root_segments.tif.tif"

        segmentation_fpath = os.path.join(base_dirpath, segmentation_name)

        volume = volread(segmentation_fpath)
        transposed = np.transpose(volume, axes=(1, 2, 0))

        segmentation = transposed.view(Segmentation3D)
        raw_stack = self.get_stack_by_series_name(name)
        measure_stack = zoom_to_match_scales(raw_stack)

        file_info_fpath = os.path.join(base_dirpath, file_info_fname)
        fids_to_rids = self.load_file_data(file_info_fpath)

        return segmentation, measure_stack, fids_to_rids


def find_single_cell_region_3D(segmentation, rid):
    isolated_space = np.zeros(segmentation.shape, dtype=np.int16)
    isolated_space[np.where(segmentation == rid)] = 1
    single_reconstructed_cell = binary_erosion(binary_dilation(isolated_space))

    return single_reconstructed_cell


def measure_single_region(segmentation, measure_stack, rid, erosion_rounds):

    single_cell_mask = find_single_cell_region_3D(segmentation, rid)

    for _ in range(erosion_rounds):
        single_cell_mask = binary_erosion(single_cell_mask)

    masked_region = single_cell_mask.astype(int) * measure_stack
    total_signal = np.sum(masked_region)
    volume = np.sum(single_cell_mask)
    centroid = [int(np.mean(a)) for a in np.where(single_cell_mask)]

    return int(total_signal), volume, centroid


def measure_from_set(segmentation, rids, measure_stack, erosion_rounds=0):
    measurements = []
    for rid in rids:
        total_signal, volume, centroid = measure_single_region(
            segmentation,
            measure_stack,
            rid,
            erosion_rounds
        )

        measurement = {
            'total_signal': total_signal,
            'volume': volume,
            'centroid': centroid
        }

        measurements.append(measurement)

    return measurements


def measure_all_files(s, f, m):
    measurements_by_file = []
    for fid, rids in f.items():
        logging.info(f"Measuring file {fid}")
        measurements = measure_from_set(s, rids, m, erosion_rounds=5)
        df = pd.DataFrame(measurements)
        df['mean_signal'] = df['total_signal'] / df['volume']
        df['file'] = fid
        measurements_by_file.append(df)

    return pd.concat(measurements_by_file)


def get_disc(R=5):
    x, y = np.ogrid[-R:R, -R:R]
    disc = x**2 + y**2 < R**2

    return disc


def vis_colour(points_list, dim, R=5):
    rdim, cdim, _ = dim
    vis = np.zeros((rdim, cdim, 3), dtype=np.uint8)

    def get_col(n):
        r = int(255 * n/len(points_list))
        g = 255 - r
        return r, g, 0

    for n, p in enumerate(points_list):
        r, c, z = p
        vis[r-R:r+R,c-R:c+R] = get_col(n)

    return vis.view(Image)


def closest_point(available_points, p):
    v = np.array(list(available_points)) - p
    sq_dists = np.sum(v * v, axis=1)
    closest_index = np.argmin(sq_dists)
    sq_d = sq_dists[closest_index]
    return list(available_points)[closest_index], sq_d


def find_end(points_list):
    available_points = set(points_list)

    p = points_list[0]

    while(len(available_points) > 1):
        available_points.remove(tuple(p))
        p_next, sq_d = closest_point(available_points, p)
        if sq_d > 100000:
            return p
        p = p_next

    return p


def find_end_min_c(points_list):
    return sorted(points_list, key=lambda p: p[1])[0]


def order_points(coords_list, p_start=None):
    available_points = set(coords_list)
    if p_start is None:
        p_start = coords_list[0]
    p = p_start

    available_points.remove(p)

    ordered_points = []
    while len(available_points):
        ordered_points.append(p)
        p_next, _ = closest_point(available_points, p)
        p = p_next
        available_points.remove(p)
    ordered_points.append(p)

    return ordered_points


def reorder_points(point_centroids):
    p_start = find_end_min_c(point_centroids)
    ordered_points = order_points(point_centroids, p_start)
    return ordered_points


def get_rank_lookup(centroids_dict, cnid):
    raw_cns_to_points_tuple_list = {cn: tuple(ast.literal_eval(cn)) for cn in centroids_dict[cnid]}
    ordered_points = reorder_points(raw_cns_to_points_tuple_list.values())
    p_tuple_to_rank = {p: ordered_points.index(p) for p in raw_cns_to_points_tuple_list.values()}
    rank_lookup = {raw_cn: p_tuple_to_rank[p] for raw_cn, p in raw_cns_to_points_tuple_list.items()}
    return rank_lookup


def add_ranks(df):
    centroids_dict = df.groupby('file')['centroid'].apply(list).to_dict()
    rank_lookups = {fid: get_rank_lookup(centroids_dict, fid) for fid in centroids_dict}

    def get_rank(file, centroid):
        return rank_lookups[file][centroid]

    df['rank'] = df.apply(lambda row: get_rank(row['file'], row['centroid']), axis=1)
