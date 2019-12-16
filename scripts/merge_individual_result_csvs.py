import os
from pathlib import Path

import click

import pandas as pd


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

    merged_df = pd.concat(all_dfs)

    merged_df.to_csv(output_csv_fpath, index=False)


if __name__ == "__main__":
    main()
