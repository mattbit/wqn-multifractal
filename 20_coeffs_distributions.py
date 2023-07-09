import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import stats

plt.style.use("./publication.mplstyle")


# Load data and normalize
eeg = np.load("data/eeg-denoise-net/EEG_all_epochs.npy")
eog = np.load("data/eeg-denoise-net/EOG_all_epochs.npy")
emg = np.load("data/eeg-denoise-net/EMG_all_epochs.npy")

n_eeg = (eeg - eeg.mean(axis=-1).reshape(-1, 1)) / eeg.std(axis=-1).reshape(-1, 1)
n_eog = (eog - eog.mean(axis=-1).reshape(-1, 1)) / eog.std(axis=-1).reshape(-1, 1)
n_emg = (emg - emg.mean(axis=-1).reshape(-1, 1)) / emg.std(axis=-1).reshape(-1, 1)


# Wavelet transform
max_levels = 4
eeg_coeffs = pywt.wavedec(n_eeg, "db3", level=max_levels, axis=-1, mode="periodization")
eog_coeffs = pywt.wavedec(n_eog, "db3", level=max_levels, axis=-1, mode="periodization")
emg_coeffs = pywt.wavedec(n_emg, "db3", level=max_levels, axis=-1, mode="periodization")

scales = [n for n in range(len(eeg_coeffs), 0, -1)]
level_names = [f"Level $c_{n}$" for n in scales]


# Prepare figure

num_levels = 5
xlim = (-8, 8)

fig = plt.figure(figsize=(7.25, 2.8), constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.02)

figA, figB = fig.subfigures(ncols=2, width_ratios=(2, 1))

# Figure B
figB.suptitle("B", x=0.21, y=1.03)
ax = figB.subplots()

(leeg,) = ax.plot(scales, [c.var() for c in eeg_coeffs], marker=".", label="EEG")
(leog,) = ax.plot(scales, [c.var() for c in eog_coeffs], marker=".", label="EOG")
(lemg,) = ax.plot(scales, [c.var() for c in emg_coeffs], marker=".", label="EMG")
ax.set_yscale("log")
ax.set_ylabel("Variance of coefficients", labelpad=-2)
ax.set_xlabel("Coefficient scale $j$")
ax.legend()

figA.suptitle("A", x=0.045, y=1.03)
axes = figA.subplots(
    ncols=num_levels,
    sharex=True,
    nrows=3,
)

num_bins = 101
bin_range = (-10, 10)
n = 0
for cs_eeg, cs_eog, cs_emg, name in zip(
    eeg_coeffs[:num_levels],
    eog_coeffs[:num_levels],
    emg_coeffs[:num_levels],
    level_names[:num_levels],
):
    vals, bins, _ = axes[0, n].hist(
        cs_eeg.reshape(-1),
        bins=num_bins,
        range=bin_range,
        density=True,
        alpha=0.7,
        fc=leeg.get_color(),
    )

    axes[0, n].plot(
        bins,
        stats.norm.pdf(bins, loc=0, scale=cs_eeg.std()),
        c="k",
        ls="--",
        lw=0.5,
        label=f"σ = {cs_eeg.std():.2f}",
    )
    ymin, ymax = axes[0, n].get_ylim()
    axes[0, n].set_ylim(ymin, ymax * 1.2)
    axes[0, n].text(
        0.05, 0.98, f"σ = {cs_eeg.std():.2f}", va="top", transform=axes[0, n].transAxes
    )
    # axes[0, n].legend()

    vals, bins, _ = axes[1, n].hist(
        cs_eog.reshape(-1),
        # fc="#2D9CDB",
        bins=num_bins,
        range=bin_range,
        density=True,
        alpha=0.7,
        fc=leog.get_color(),
    )

    axes[2, n].hist(
        cs_emg.reshape(-1),
        # fc="#2D9CDB",
        bins=num_bins,
        range=bin_range,
        density=True,
        alpha=0.7,
        fc=lemg.get_color(),
    )

    axes[-1, n].set_xlabel(name)
    axes[0, n].set_yticks([])
    axes[1, n].set_yticks([])
    axes[2, n].set_yticks([])

    n += 1

axes[0, 0].set_ylabel("EEG")
axes[1, 0].set_ylabel("EOG")
axes[2, 0].set_ylabel("EMG")

axes[0, 0].set_xticks([-5, 0, 5])
axes[1, 0].set_xticks([-5, 0, 5])
axes[2, 0].set_xticks([-5, 0, 5])
axes[0, 0].set_xlim(*xlim)
axes[1, 0].set_xlim(*xlim)
axes[2, 0].set_xlim(*xlim)


fig.savefig("output/fig_coeffs_distribution.pdf")
