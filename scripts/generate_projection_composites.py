import os
import ast
import logging

import click

import pandas as pd
import numpy as np

from dtoolbioimage import Image, Image3D, scale_to_uint8, ImageDataSet, zoom_to_match_scales
from dtoolbioimage.segment import Segmentation, Segmentation3D

from PIL import Image as pilImage, ImageDraw, ImageFont

from stacktools.cache import fn_caching_wrapper

from stacktools.utils import (
    calculate_hdome,
    filter_segmentation_by_regions,
    centroid_project,
)

from stacktools.data import FCADataLoader, FCARootData


def create_region_label_image(cdim, centroid_by_rid):
    label_canvas = np.zeros((100, cdim), dtype=np.float32)
    label_canvas_im = pilImage.fromarray(label_canvas)
    label_draw = ImageDraw.ImageDraw(label_canvas_im)
    fnt_size = 10
    fnt = ImageFont.truetype('Microsoft Sans Serif.ttf', fnt_size)

    for rid, cn in centroid_by_rid.items():
        label_draw.text((cn[1]-5, 25), str(rid), font=fnt, fill=1.0)

    label_array = np.array(label_canvas_im)

    return label_array


def create_and_join_file_projections(centroid_by_rid, trimmed_segmentation, stacks):
    projs = [centroid_project(s, trimmed_segmentation, centroid_by_rid) for s in stacks]

    rmin = min(np.where(projs[0])[0])
    rmax = max(np.where(projs[0])[0])

    cdim = projs[0].shape[1]

    label_array = create_region_label_image(cdim, centroid_by_rid)

    return np.vstack([p[rmin:rmax,:] for p in projs] + [label_array]).view(Image)


def file_position_image(wall_stack, file_coords):

    c = int(file_coords[1].mean())

    file_mask_stack = scale_to_uint8(wall_stack.copy())
    file_mask_stack[file_coords] = 255

    return file_mask_stack[:,c,:].T.view(Image)


def file_position_image_r(wall_stack, file_coords):

    r = int(file_coords[0].mean())

    print(f"r = {r}")

    file_mask_stack = scale_to_uint8(wall_stack.copy())
    file_mask_stack[file_coords] = 255

    return file_mask_stack[r,:,:].T.view(Image)


def all_file_coords(segmentation, rids):
    coord_blocks = [np.where(segmentation == rid) for rid in rids]

    rs, cs, zs = list(zip(*coord_blocks))

    rr = np.concatenate(rs)
    cc = np.concatenate(cs)
    zz = np.concatenate(zs)

    return rr, cc, zz


def file_summary_image(trimmed_segmentation, file, wall_stack, stacks, root_name, centroid_by_rid):

    rids = centroid_by_rid.keys()
    coords = all_file_coords(trimmed_segmentation, rids)
    fpir = file_position_image_r(wall_stack, coords)
    fpi = file_position_image(wall_stack, coords)
    rdim, cdim = fpi.shape
    markme = np.pad(fpi, ((0, 0), (512, 0))).view(Image)

    pili = pilImage.fromarray(markme)
    draw = ImageDraw.ImageDraw(pili)
    fnt_size = 18
    fnt = ImageFont.truetype('Microsoft Sans Serif.ttf', fnt_size)

    label = f"{root_name} - file {file}"

    draw.text((10, 30), label, font=fnt, fill=255)

    parrray = np.array(pili)
    return np.vstack([
        parrray,
        fpir,
        scale_to_uint8(create_and_join_file_projections(centroid_by_rid, trimmed_segmentation, stacks)
    )]).view(Image)


def get_centroids_by_rid(df, fid):
    centroid_by_rid_raw = df[df.file_id == fid].set_index('region_id').cell_centroid.to_dict()
    centroid_by_rid = {rid: ast.literal_eval(c) for rid, c in centroid_by_rid_raw.items()}

    return centroid_by_rid


@fn_caching_wrapper
def get_stack_by_name(ids_uri, root_name, channel=0):

    ids = ImageDataSet.from_uri(ids_uri)
    name_lookup = dict(ids.get_image_series_name_pairs())
    series_name = name_lookup[root_name]

    raw_stack = ids.get_stack(root_name, series_name, channel=channel)
    zoomed_stack = zoom_to_match_scales(raw_stack)

    return zoomed_stack


@click.command()
@click.argument('image_ds_uri')
@click.argument('segmentation_dirpath')
@click.argument('root_data_basepath')
@click.option('--root-name', default="fca3_FLCVenus_root2")
def main(image_ds_uri, segmentation_dirpath, root_data_basepath, root_name):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("generate_projection_composites")
    logging.getLogger("stacktools.cache").setLevel(level=logging.DEBUG)

    logger.info(f"Loading root {root_name}")
    loader = FCADataLoader(image_ds_uri, root_data_basepath, segmentation_dirpath)
    rootdata = loader.load_root(root_name)

    logger.info(f"Filtering segmentation for regions in files")
    trimmed_segmentation = filter_segmentation_by_regions(
        rootdata.segmentation, rootdata.regions_in_files
    )

    stacks = [
        rootdata.venus_stack,
        rootdata.denoised_venus_stack,
        rootdata.hdome_venus_stack
    ]

    for fid in rootdata.files:
        fsm = file_summary_image(
            trimmed_segmentation, fid, rootdata.wall_stack, stacks, root_name, rootdata.cell_centroids(fid)
            )
        fsm.save(f"fsm-{root_name}-file{fid}.png")


# @click.command()
# @click.argument('image_ds_uri')
# @click.argument('segmentation_dirpath')
# @click.argument('root_data_basepath')
# @click.option('--root-name', default="fca3_FLCVenus_root2")
# def main(image_ds_uri, segmentation_dirpath, root_data_basepath, root_name):
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger("generate_projection_composites")
#     logging.getLogger("stacktools.cache").setLevel(level=logging.DEBUG)

#     logger.info("Loading stacks")
#     venus_stack = get_stack_by_name(image_ds_uri, root_name)
#     wall_stack = get_stack_by_name(image_ds_uri, root_name, channel=1)

#     root_data_fpath = os.path.join(root_data_basepath, f"{root_name}-spherefit.csv")
#     logger.info(f"Loading root data from {root_data_fpath}")
#     df = pd.read_csv(root_data_fpath)
#     logger.info(f"Loading segmentation from {segmentation_dirpath}")
#     segmentation = get_segmentation(segmentation_dirpath, root_name).view(Segmentation3D)
#     logger.info(f"Filtering segmentation for regions in files")
#     trimmed_segmentation = filter_segmentation_by_regions(segmentation, df.region_id)

#     logger.info("Generating denoised measurement stack")
#     denoised_venus_stack = denoise_tv_chambolle_f32(venus_stack, weight=0.01)
#     logger.info("Calculating background subtracted stack")
#     hdome_venus_stack = calculate_hdome(denoised_venus_stack)

#     stacks = [venus_stack, denoised_venus_stack, hdome_venus_stack]

#     for fid in set(df.file_id):
#         centroid_by_rid = get_centroids_by_rid(df, fid)
#         fsm = file_summary_image(trimmed_segmentation, fid, wall_stack, stacks, root_name, centroid_by_rid)
#         fsm.save(f"fsm-{root_name}-file{fid}.png")


if __name__ == "__main__":
    main()
