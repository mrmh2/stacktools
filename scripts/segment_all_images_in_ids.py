import logging
import pathlib

import click

from dtoolbioimage import ImageDataSet, zoom_to_match_scales
from dtoolbioimage.segment import sitk_watershed_segmentation, filter_segmentation_by_size


def derive_output_filename(output_dirpath, image_name, series_name):

    return output_dirpath/(image_name + '_segmentation.tif')


def segment_image_from_dataset(imageds, image_name, series_name, wall_channel, output_filename, level=0.664, nsegments=200):

    logging.info("Loading wall stack")
    wall_stack = imageds.get_stack(image_name, series_name, 0, wall_channel)
    logging.info("Adjusting scales")
    zoomed_wall_stack = zoom_to_match_scales(wall_stack)
    logging.info("Segmenting image")
    segmentation = sitk_watershed_segmentation(zoomed_wall_stack, level=level)
    logging.info(f"Filtering segmentation by cell size, max {nsegments} cells")
    filtered_segmentation = filter_segmentation_by_size(segmentation, nsegments)
    filtered_segmentation.save(output_filename)


def segment_all_images_in_dataset(imageds, channel, output_dirpath):
    for image_name in imageds.get_image_names():
        for series_name in imageds.get_series_names(image_name):
            output_filename = derive_output_filename(output_dirpath, image_name, series_name)
            segment_image_from_dataset(imageds, image_name, series_name, channel, output_filename)


@click.command()
@click.argument('dataset_uri')
@click.argument('output_dirpath')
def main(dataset_uri, output_dirpath):

    logging.basicConfig(level=logging.INFO)

    imageds = ImageDataSet(dataset_uri)
    output_dirpath = pathlib.Path(output_dirpath)

    for image_name in imageds.get_image_names():
        for series_name in imageds.get_series_names(image_name):
            logging.info(f"Processing {image_name} {series_name}")
            level = 0.3
            nsegments = 5000
            wall_channel = 1
            output_filename = f'{image_name}_{series_name}_L{level}.tif'
            output_fpath = output_dirpath / output_filename
            segment_image_from_dataset(
                imageds,
                image_name,
                series_name,
                wall_channel,
                output_filename,
                level,
                nsegments
            )



if __name__ == "__main__":
    main()