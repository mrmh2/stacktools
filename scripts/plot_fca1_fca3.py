import click

import pandas as pd

import matplotlib.pyplot as plt


@click.command()
@click.argument('fca1_csv_fpath')
@click.argument('fca3_csv_fpath')
def main(fca1_csv_fpath, fca3_csv_fpath):

    df_fca1 = pd.read_csv(fca1_csv_fpath)
    df_fca3 = pd.read_csv(fca3_csv_fpath)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    # ax.set_title("Intensity")
    # ax.set_xlabel("Mean intensity")
    ax2 = fig.add_subplot(2, 1, 2)


    ax1.hist(df_fca1.fitted_measurement, bins=35, range=[4.5, 22])
    ax1.set_title("FCA1")
    ax2.hist(df_fca3.fitted_measurement, bins=35, range=[4.5, 7.5])
    ax2.set_title("FCA3")

    plt.suptitle("FCA1 vs FCA3 (sphere fitted)")
    plt.show()


if __name__ == "__main__":
    main()
