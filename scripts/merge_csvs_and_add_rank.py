import os
from pathlib import Path

import click

import pandas as pd


import ast

import numpy as np


def closest_point(available_points, p):
    v = np.array(list(available_points)) - p
    sq_dists = np.sum(v * v, axis=1)
    closest_index = np.argmin(sq_dists)
    sq_d = sq_dists[closest_index]
    return list(available_points)[closest_index], sq_d


def find_end(points_list):
    available_points = set(points_list)

    p = points_list[0]

    while(len(available_points) > 1):
        available_points.remove(tuple(p))
        p_next, sq_d = closest_point(available_points, p)
        if sq_d > 100000:
            return p
        p = p_next

    return p


def find_end_min_c(points_list):
    return sorted(points_list, key=lambda p: p[1])[0]


def order_points(coords_list, p_start=None):
    available_points = set(coords_list)
    if p_start is None:
        p_start = coords_list[0]
    p = p_start

    available_points.remove(p)

    ordered_points = []
    while len(available_points):
        ordered_points.append(p)
        p_next, _ = closest_point(available_points, p)
        p = p_next
        available_points.remove(p)
    ordered_points.append(p)

    return ordered_points


def reorder_points(point_centroids):
    p_start = find_end_min_c(point_centroids)
    ordered_points = order_points(point_centroids, p_start)
    return ordered_points


def get_rank_lookup(centroids_dict, cnid):
    raw_cns_to_points_tuple_list = {cn: tuple(ast.literal_eval(cn)) for cn in centroids_dict[cnid]}
    ordered_points = reorder_points(raw_cns_to_points_tuple_list.values())
    p_tuple_to_rank = {p: ordered_points.index(p) for p in raw_cns_to_points_tuple_list.values()}
    rank_lookup = {raw_cn: p_tuple_to_rank[p] for raw_cn, p in raw_cns_to_points_tuple_list.items()}
    return rank_lookup


def add_ranks(df):
    centroids_dict = df.groupby('file_id')['segmented_cell_centroid'].apply(list).to_dict()
    rank_lookups = {fid: get_rank_lookup(centroids_dict, fid) for fid in centroids_dict}

    def get_rank(file, centroid):
        return rank_lookups[file][centroid]

    df['rank'] = df.apply(lambda row: get_rank(row['file_id'], row['segmented_cell_centroid']), axis=1)


@click.command()
@click.argument('individual_results_dirpath')
@click.argument('output_csv_fpath')
def main(individual_results_dirpath, output_csv_fpath):

    prefix = "fca3_FLCVenus_root"
    def fpath_to_root_number(fpath):
        basename = os.path.basename(fpath)
        return int(basename[len(prefix):].split('-')[0])

    def load_and_annotate_df(fpath):
        df = pd.read_csv(fpath)
        df['root'] = fpath_to_root_number(fpath)
        return df

    all_dfs = map(load_and_annotate_df, Path(individual_results_dirpath).iterdir())

    dfs_with_rank = []
    for df in all_dfs:
        add_ranks(df)
        dfs_with_rank.append(df)


    merged_df = pd.concat(dfs_with_rank)

    merged_df.to_csv(output_csv_fpath, index=False, float_format='%.3f')


if __name__ == "__main__":
    main()
