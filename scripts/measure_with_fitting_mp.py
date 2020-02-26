import logging

# import multiprocessing as mp
from pathos.multiprocessing import ProcessPool

import click

import pandas as pd

from stacktools import DataLoader
from stacktools.measure import measure_by_obsphere_fit_brute


def measure_single_region(segmentation, measure_stack, rid):

    logging.info(f"Measuring region {rid}")
    m, p_sphere, p_cell = measure_by_obsphere_fit_brute(segmentation, measure_stack, rid)

    measurement = {
        'sphere_fit_centroid': p_sphere,
        'cell_centroid': p_cell,
        'fitted_measurement': m,
        'region_id': rid
    }

    return measurement


def measure_set_of_regions(segmentation, measure_stack, region_list):

    measurements = [
        measure_single_region(segmentation, measure_stack, rid)
        for rid in region_list
    ]

    return pd.DataFrame(measurements)


@click.command()
@click.argument('ids_uri')
@click.argument('seg_dirpath')
@click.option('--root-name', default="fca3_FLCVenus_root2")
def main(ids_uri, seg_dirpath, root_name):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )

    logging.info("Initialising dataloader")
    dl = DataLoader(ids_uri, seg_dirpath)

    logging.info(f"Loading root {root_name}")
    segmentation, venus_stack, files = dl.load_by_name(root_name)

    def measure_fid(fid):
        df = measure_set_of_regions(segmentation, venus_stack, files[fid])
        df['file_id'] = fid
        return df

    pool = ProcessPool(processes=2)

    results = pool.map(measure_fid, files)

    df_all = pd.concat(results)
    df_all.to_csv(f"{root_name}-spherefit.csv", index=False)


if __name__ == "__main__":
    main()
