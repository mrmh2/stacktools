import os
import logging

import click

from stacktools.data import FCADataSetLoader
from stacktools.vis import create_file_projection_composite


@click.command()
@click.argument('image_ds_uri')
@click.argument('seg_ds_uri')
@click.argument('root_data_basepath')
@click.argument('output_basedir')
@click.option('--root-name', default="fca3_FLCVenus_root2")
def main(image_ds_uri, seg_ds_uri, root_data_basepath, output_basedir, root_name):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("generate_projection_composites")
    logging.getLogger("stacktools.cache").setLevel(level=logging.DEBUG)

    logger.info(f"Loading root {root_name}")
    loader = FCADataSetLoader(image_ds_uri, seg_ds_uri, root_data_basepath)
    rootdata = loader.load_root(root_name)

    for fid in rootdata.files:
        fsm = create_file_projection_composite(rootdata, fid)
        fname = f"composite-{root_name}-file{fid}.png"
        fpath = os.path.join(output_basedir, fname)
        fsm.save(fpath)


if __name__ == "__main__":
    main()
