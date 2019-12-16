import click

import pandas as pd

import matplotlib.pyplot as plt


@click.command()
@click.argument('measurements_csv_fpath')
@click.argument('unfitted_csv_fpath')
def main(measurements_csv_fpath, unfitted_csv_fpath):

    df = pd.read_csv(measurements_csv_fpath)
    df_unfitted = pd.read_csv(unfitted_csv_fpath)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    # ax.set_title("Intensity")
    # ax.set_xlabel("Mean intensity")
    ax2 = fig.add_subplot(2, 1, 2)


    ax1.hist(df.fitted_measurement, bins=30, range=[4.5, 7.5])
    ax1.set_title("Sphere fitted")
    ax2.hist(df_unfitted.mean_signal, bins=30, range=[4.5, 7.5])
    ax2.set_title("Whole cell mean")

    plt.suptitle("Whole-cell mean vs. sphere fitted intensity")
    plt.show()


if __name__ == "__main__":
    main()
