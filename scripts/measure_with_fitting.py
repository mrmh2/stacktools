import logging

import click

import pandas as pd

from stacktools import DataLoader
from stacktools.measure import measure_by_obsphere_fit


def measure_set_of_regions(segmentation, measure_stack, rids):

    measurements = []
    for rid in rids:
        m, p = measure_by_obsphere_fit(segmentation, measure_stack, rid)

        if m is not 0:
            measurement = {
                'mean_sphere_signal': m,
                'sphere_centroid': p
            }
            
            measurements.append(measurement)

    return measurements


def measure_file(segmentation, measure_stack, files, fid):

    measurements = measure_set_of_regions(segmentation, measure_stack, files[fid])
    df = pd.DataFrame(measurements)
    df['file'] = fid
    
    return df

@click.command()
@click.argument('ids_uri')
@click.argument('seg_dirpath')
@click.argument('root_name')
def main(ids_uri, seg_dirpath, root_name):

    logging.basicConfig(level=logging.INFO)

    dl = DataLoader(ids_uri, seg_dirpath)

    segmentation, venus_stack, files = dl.load_by_name(root_name)
    logging.info("Loaded data")

    dfs = []
    for fid in list(files.keys())[:2]:
        logging.info(f"Measuring file {fid}")
        df = measure_file(segmentation, venus_stack, files, fid)
        # measurements = measure_set_of_regions(segmentation, venus_stack, files[fid])
        # df = pd.DataFrame(measurements)
        # df['file'] = fid

        dfs.append(df)
    
    root_df = pd.concat(dfs, ignore_index=True)

    root_df['root'] = 2

    print(root_df.to_csv(index=False))


if __name__ == "__main__":
    main()
