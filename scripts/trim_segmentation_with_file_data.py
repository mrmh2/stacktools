import logging

import click

import numpy as np

from stacktools import DataLoader


def trim_segmentation(segmentation, files):
    trimmed_segmentation = segmentation.copy()

    rids_in_files = set(sum(files.values(), []))

    rids_not_in_files = segmentation.labels - rids_in_files

    for rid in rids_not_in_files:
        trimmed_segmentation[np.where(trimmed_segmentation == rid)] = 0

    return trimmed_segmentation


@click.command()
@click.argument('image_ds_uri')
@click.argument('segmentation_dirpath')
@click.argument('output_dirpath')
@click.option('--root-name', default="fca3_FLCVenus_root2")
def main(image_ds_uri, segmentation_dirpath, output_dirpath, root_name):

    logging.basicConfig(level=logging.INFO)

    logging.info("Initialising data loader")
    dl = DataLoader(image_ds_uri, segmentation_dirpath)
    logging.info(f"Loading root {root_name}")
    segmentation, venus_stack, files = dl.load_by_name(root_name)

    trimmed_segmentation = trim_segmentation(segmentation, files)

    output_fname = f"{root_name}-segmentation-trimmed.tif"
    trimmed_segmentation.save(output_fname)


if __name__ == "__main__":
    main()
