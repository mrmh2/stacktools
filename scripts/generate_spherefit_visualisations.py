import os
import click

import numpy as np
from dtoolbioimage import Image

from stacktools.data import FCADataSetLoader, FCARootData
from stacktools.vis import centroid_project, create_annotated_file_projection


@click.command()
@click.argument('image_ds_uri')
@click.argument('seg_ds_uri')
@click.argument('root_data_dirpath')
@click.argument('output_base_dirpath')
@click.option('--root-name', default="fca3_FLCVenus_root2")
def main(image_ds_uri, seg_ds_uri, root_data_dirpath, output_base_dirpath, root_name):

    loader = FCADataSetLoader(image_ds_uri, seg_ds_uri, root_data_dirpath)
    rootdata = loader.load_root(root_name)

    sections = [
        create_annotated_file_projection(rootdata, fid, rootdata.denoised_venus_stack)
        for fid in rootdata.files
    ]

    composite = np.vstack(sections).view(Image)

    fname = f"{root_name}-spherefit-allfiles.png"
    fpath = os.path.join(output_base_dirpath, fname)
    composite.save(fpath)


if __name__ == "__main__":
    main()
