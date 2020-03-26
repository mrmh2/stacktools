import os
import ast
import logging

from dtoolbioimage.segment import Segmentation3D
from pathos.multiprocessing import ProcessPool

import click

import pandas as pd


from stacktools import DataLoader
from stacktools.measure import measure_by_obsphere_fit_brute, multi_measure
from stacktools.data import get_stack_by_name, get_masked_venus_stack
from stacktools.utils import get_segmentation, filter_segmentation_by_regions


def measure_single_region(segmentation, measure_stack, rid):

    logging.info(f"Measuring region {rid}")

    measurement = multi_measure(segmentation, measure_stack, rid)
    measurement['region_id'] = rid

    return measurement


def measure_set_of_regions(segmentation, measure_stack, region_list):

    measurements = [
        measure_single_region(segmentation, measure_stack, rid)
        for rid in region_list
    ]

    return pd.DataFrame(measurements)


def get_centroids_by_rid(df, fid):
    centroid_by_rid_raw = df[df.file_id == fid].set_index('region_id').cell_centroid.to_dict()
    centroid_by_rid = {rid: ast.literal_eval(c) for rid, c in centroid_by_rid_raw.items()}

    return centroid_by_rid



@click.command()
@click.argument('image_ds_uri')
@click.argument('segmentation_dirpath')
@click.argument('root_data_basepath')
@click.option('--root-name', default="fca3_FLCVenus_root2")
def main(image_ds_uri, segmentation_dirpath, root_name, root_data_basepath):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )

    logger = logging.getLogger('measure_mutiple')
    logger.setLevel(logging.DEBUG)

    logger.info("Loading stacks")
    masked_venus_stack = get_masked_venus_stack(image_ds_uri, root_name)
    root_data_fpath = os.path.join(root_data_basepath, f"{root_name}-spherefit.csv")
    logger.info(f"Loading root data from {root_data_fpath}")
    root_df = pd.read_csv(root_data_fpath)
    logger.info(f"Loading segmentation from {segmentation_dirpath}")
    segmentation = get_segmentation(segmentation_dirpath, root_name).view(Segmentation3D)
    logger.info(f"Filtering segmentation for regions in files")
    trimmed_segmentation = filter_segmentation_by_regions(segmentation, root_df.region_id)

    file_ids = set(root_df.file_id)

    def measure_fid(fid):
        rids = list(root_df[root_df.file_id == fid].region_id)
        measure_df = measure_set_of_regions(trimmed_segmentation, masked_venus_stack, rids[:3])
        measure_df['file_id'] = fid
        return measure_df

    pool = ProcessPool(processes=3)

    results = pool.map(measure_fid, file_ids)
    df_all = pd.concat(results)
    df_all.to_csv(f"{root_name}-multimeasure-masked.csv", float_format='%.3f', index=False)


if __name__ == "__main__":
    main()
