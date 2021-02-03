import numpy as np

from PIL import Image as pilImage, ImageDraw, ImageFont

from skimage.morphology import erosion, dilation
from skimage.draw import circle_perimeter

from dtoolbioimage import Image, scale_to_uint8

from stacktools.measure import measure_properties


def scale_factors(array):
    scaled = array.astype(np.float32)
    return scaled.min(), scaled.max() - scaled.min()


def centroid_project(stack, segmentation, centroids_by_rid):
    canvas = np.zeros((512, 1024), dtype=np.float32)

    coords_vals = []
    for rid, centroid in centroids_by_rid.items():
        r, c, z = centroid
        closed_region = erosion(dilation(segmentation[:, :, int(z)] == rid))
        rr, cc = np.where(closed_region)
        coords_vals.append(((rr, cc), stack[rr, cc, int(z)]))

    smin, srange = scale_factors(np.concatenate([v for c, v in coords_vals]))
    for (rr, cc), val in coords_vals:
        canvas[rr, cc] = ((val - smin) / srange)

    return canvas.view(Image)


def variable_z_project(stack, segmentation, rid_to_z):
    canvas = np.zeros((512, 1024), dtype=np.float32)

    coords_vals = []
    for rid, z in rid_to_z.items():
        closed_region = erosion(dilation(segmentation[:, :, int(z)] == rid))
        rr, cc = np.where(closed_region)
        coords_vals.append(((rr, cc), stack[rr, cc, int(z)]))

    smin, srange = scale_factors(np.concatenate([v for c, v in coords_vals]))
    for (rr, cc), val in coords_vals:
        canvas[rr, cc] = ((val - smin) / srange)

    return canvas.view(Image)


def file_position_image_transverse(wall_stack, coords):
    c = int(coords[1].mean())
    file_mask_plane = wall_stack[:, c, :]
    plane_colour = scale_and_convert_to_rgb(file_mask_plane)
    section_idxs = np.where(coords[1] == c)
    section_coords = coords[0][section_idxs], coords[2][section_idxs]
    plane_colour[section_coords] = 0, 255, 255

    return np.transpose(plane_colour, (1, 0, 2)).view(Image)


def file_position_image_lateral(wall_stack, coords):
    r= int(coords[0].mean())
    file_mask_plane = wall_stack[r, :, :]
    plane_colour = scale_and_convert_to_rgb(file_mask_plane)
    section_idxs = np.where(coords[0] == r)
    section_coords = coords[1][section_idxs], coords[2][section_idxs]
    plane_colour[section_coords] = 0, 255, 255

    return np.transpose(plane_colour, (1, 0, 2)).view(Image)


def file_position_image(wall_stack, file_coords):

    c = int(file_coords[1].mean())

    file_mask_stack = scale_and_convert_to_rgb(wall_stack.copy())
    file_mask_stack[file_coords] = 0, 255, 255 

    return file_mask_stack[:, c, :, :].T.view(Image)


def file_position_image_r(wall_stack, file_coords):

    r = int(file_coords[0].mean())

    file_mask_stack = scale_to_uint8(wall_stack.copy())
    file_mask_stack[file_coords] = 255

    return file_mask_stack[r, :, :].T.view(Image)


def find_projection_row_bounds(projection):
    row_coords = np.nonzero(projection)[0]
    return min(row_coords), max(row_coords)


def trim_image_byrow(im):
    rmin, rmax = find_projection_row_bounds(im)
    return im[rmin:rmax, :, :]


def scale_and_convert_to_rgb(im):
    scaled = scale_to_uint8(im)

    if len(im.shape) == 2:
        return np.dstack(3 * [scaled])

    return im


def annotate_with_circles(im, cn_list, radius):
    for r, c in cn_list:
        rr, cc = circle_perimeter(r, c, radius)
        im[rr, cc] = 255, 255, 0


def create_annotated_file_projection(rootdata, fid, stack, radius=15):
    rid_to_z_sp = {
        rid: cn[2]
        for rid, cn in rootdata.sphere_centroids(fid).items()
    }
    projection = variable_z_project(stack, rootdata.segmentation, rid_to_z_sp)
    proj_colour = scale_and_convert_to_rgb(projection)
    cn_list = [(r, c) for r, c, z in rootdata.sphere_centroids(fid).values()]
    annotate_with_circles(proj_colour, cn_list, radius)
    trimmed = trim_image_byrow(proj_colour)

    return trimmed


def create_and_join_all_annotated_projections(rootdata, fid):
    stacks = rootdata.venus_stack, rootdata.denoised_venus_stack, rootdata.hdome_venus_stack
    annotated_projections = [
        create_annotated_file_projection(rootdata, fid, stack)
        for stack in stacks
    ]

    joined_projections = np.vstack(annotated_projections)
    return joined_projections


def annotate_with_text(im, text):
    pilim = pilImage.fromarray(im)
    draw = ImageDraw.ImageDraw(pilim)
    fnt = ImageFont.truetype('Microsoft Sans Serif.ttf', size=18)
    draw.text((10, 30), text, font=fnt, fill=(255, 255, 255))

    return np.array(pilim)


def create_header(rootdata, fid):
    coords = rootdata.file_coords(fid)
    fpi_transverse = file_position_image_transverse(
        rootdata.wall_stack, coords)
    cdim = rootdata.wall_stack.shape[1]
    cpad = cdim - fpi_transverse.shape[1]
    padded = np.pad(fpi_transverse, ((0, 0), (cpad, 0), (0, 0)))
    label = f"{rootdata.name} - file {fid}"
    return annotate_with_text(padded, label).view(Image)


def create_region_label_image(cdim, centroid_by_rid):
    label_canvas = np.zeros((50, cdim, 3), dtype=np.uint8)
    label_canvas_im = pilImage.fromarray(label_canvas)
    label_draw = ImageDraw.ImageDraw(label_canvas_im)
    fnt_size = 10
    fnt = ImageFont.truetype('Microsoft Sans Serif.ttf', fnt_size)

    voffset = 0
    for rid, cn in centroid_by_rid.items():
        label_draw.text((cn[1]-5, 25+voffset), str(rid), font=fnt)
        voffset = -voffset

    label_array = np.array(label_canvas_im)

    return label_array


def create_file_projection_composite(rootdata, fid):

    header = create_header(rootdata, fid)
    coords = rootdata.file_coords(fid)
    fpi_lateral = file_position_image_lateral(rootdata.wall_stack, coords)
    projections = create_and_join_all_annotated_projections(rootdata, fid)
    region_label_image = create_region_label_image(
        rootdata.wall_stack.shape[1], rootdata.cell_centroids(fid))

    return np.vstack([header, fpi_lateral, projections, region_label_image]).view(Image)


def create_colour_func(values):
    vmin = min(values)
    vmax = max(values)
    vdiff = vmax - vmin

    def colour_func(v):
        G = int(255 * ((v - vmin) / vdiff))
        R = 255 - G
        return R, G, 0

    return colour_func


def create_heatmap(sms, z):
    measurements = [
        measure_properties(sms, rid)
        for rid in sms.segmentation.labels
    ]

    intensities_by_label = {
        measurement['label']: measurement['mean_intensity']
        for measurement in measurements
    }

    rdim, cdim, _ = sms.measure_stack.shape
    heatmap = np.zeros((rdim, cdim, 3), dtype=np.uint8)
    colour_func = create_colour_func(intensities_by_label.values())

    seg_plane = sms.segmentation[:, :, z]
    for l, intensity in intensities_by_label.items():
        coords = np.where(seg_plane == l)
        heatmap[coords] = colour_func(intensity)

    return heatmap.view(Image)
