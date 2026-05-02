"""Plotting theme and helpers."""

import locale

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


def apply_theme(font_family: str = "Space Grotesk") -> None:
    """Apply the project plotting theme."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.formatter.use_locale": True,
        }
    )
