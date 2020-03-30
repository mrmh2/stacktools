import click

import numpy as np
from dtoolbioimage import Image

from stacktools.data import FCADataSetLoader, FCARootData
from stacktools.vis import centroid_project, generate_annotated_file_projection



@click.command()
@click.argument('image_ds_uri')
@click.argument('seg_ds_uri')
@click.argument('root_data_dirpath')
@click.option('--root-name', default="fca3_FLCVenus_root2")
def main(image_ds_uri, seg_ds_uri, root_data_dirpath, root_name):

    loader = FCADataSetLoader(image_ds_uri, seg_ds_uri, root_data_dirpath)
    rootdata = loader.load_root(root_name)

    sections = [
        generate_annotated_file_projection(rootdata, fid)
        for fid in rootdata.files
    ]

    composite = np.vstack(sections).view(Image)

    composite.save(f"{root_name}-composite.png")


if __name__ == "__main__":
    main()
