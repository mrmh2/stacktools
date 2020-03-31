import os
import pathlib

from collections import defaultdict

import pandas as pd
import click
import dtoolcore


def fpath_to_name(fpath):
    return os.path.splitext(fpath.name)[0]


def load_file_data(file_fpath):
    df = pd.read_csv(file_fpath, names=['rid', 'file'])
    file_lookup = pd.Series(df.file.values, index=df.rid).to_dict()
    fids_to_rids = defaultdict(list)
    for rid, fid in file_lookup.items():
        if fid is not 0:
            fids_to_rids[fid].append(rid)

    return fids_to_rids


@click.command()
@click.argument('source_dirpath')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(source_dirpath, output_base_uri, output_name):

    dirpath = pathlib.Path(source_dirpath)
    diter = pathlib.Path(source_dirpath).glob("*.dbim")

    with dtoolcore.DataSetCreator(output_name, output_base_uri) as output_ds:
        for fpath in diter:
            name = fpath_to_name(fpath)
            files = load_file_data(dirpath/f"{name}.csv")

            handle = output_ds.put_item(fpath, fpath.name)
            output_ds.add_item_metadata(handle, "files", files)


if __name__ == "__main__":
    main()
