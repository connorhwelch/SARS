import numpy as np 
import matplotlib.pyplot as plt


def plot_difference_histogram(
    corrected,
    uncorrected,
    *,
    percent=False,
    bins=100,
    range_percentile=99,
    title=None,
    ax=None
):
    """
    Plot a histogram of band differences (corrected - uncorrected).

    Parameters
    ----------
    corrected : ndarray
        Corrected band data
    uncorrected : ndarray
        Uncorrected band data
    percent : bool, optional
        If True, plot percent difference: 100 * (c - u) / u
    bins : int, optional
        Number of histogram bins
    range_percentile : float, optional
        Percentile used to limit histogram range (symmetric)
    title : str, optional
        Plot title
    ax : matplotlib axis, optional
        Existing axis to plot on

    Returns
    -------
    stats : dict
        Dictionary of summary statistics
    """

    # Compute difference
    if percent:
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = 100.0 * (corrected - uncorrected) / uncorrected
        xlabel = "Percent difference (%)"
    else:
        diff = np.array(corrected - uncorrected)
        xlabel = "Difference (corrected − uncorrected)"

    # Flatten and remove NaNs/Infs
    diff = diff.ravel()
    diff = diff[np.isfinite(diff)]

    # Robust range using percentiles
    p = np.nanpercentile(np.abs(diff), range_percentile)
    hist_range = (-p, p)

    # Create axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    # Plot histogram
    ax.hist(diff, bins=bins, range=hist_range, histtype="stepfilled", alpha=0.7)

    # Statistics
    mean = np.nanmean(diff)
    median = np.nanmedian(diff)
    std = np.nanstd(diff)
    rmse = np.sqrt(np.nanmean(diff**2))

    # Overlay statistics
    ax.axvline(mean, linestyle="--", linewidth=2, label=f"Mean = {mean:.3g}")
    ax.axvline(median, linestyle=":", linewidth=2, label=f"Median = {median:.3g}")

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel count")
    ax.legend()

    if title is not None:
        ax.set_title(title)

    stats = {
        "mean": mean,
        "median": median,
        "std": std,
        "rmse": rmse,
        "n_pixels": diff.size,
    }

    return stats