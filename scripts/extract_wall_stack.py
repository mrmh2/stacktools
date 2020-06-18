import click
import logging

from dtoolbioimage import ImageDataSet, zoom_to_match_scales


@click.command()
@click.argument('dataset_uri')
def main(dataset_uri):

    logging.basicConfig(level=logging.INFO)

    imageds = ImageDataSet(dataset_uri)

    print(imageds.all_possible_stack_tuples())
    image_name = '20200309_lhp1_W10_T14'
    series_name = 'SDB995-5_01'
    wall_channel = 1

    logging.info("Loading wall stack")
    wall_stack = imageds.get_stack(image_name, series_name, 0, wall_channel)
    logging.info("Adjusting scales")
    zoomed_wall_stack = zoom_to_match_scales(wall_stack)

    output_filename = f'{image_name}_{series_name}_wall.tif'
    zoomed_wall_stack.save(output_filename)

if __name__ == "__main__":
    main()
