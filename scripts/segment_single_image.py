import logging

from pathlib import Path


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
def main(dataset_uri):

    logging.basicConfig(level=logging.INFO)

    imageds = ImageDataSet(dataset_uri)

    print(imageds.all_possible_stack_tuples())
    image_name = '20200309_lhp1_W10_T14'
    series_name = 'SDB995-5_03'


    wall_channel = 1

    level = 0.3
    output_filename = f'{image_name}_{series_name}_L{level}.tif'
    segment_image_from_dataset(imageds, image_name, series_name, wall_channel, output_filename, level=level, nsegments=5000)
    # segment_all_images_in_dataset(imageds, wall_channel, Path('scratch/'))


if __name__ == "__main__":
    main()
