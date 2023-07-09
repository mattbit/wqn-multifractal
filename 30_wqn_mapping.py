import matplotlib.pyplot as plt
import numpy as np
import pywt

plt.style.use("./publication.mplstyle")

RED = "#fc8d62"
GREEN = "#66c2a5"
BLUE = "#8da0cb"

# %% Load data

eeg = np.load("data/eeg-denoise-net/EEG_all_epochs.npy")
eog = np.load("data/eeg-denoise-net/EOG_all_epochs.npy")
emg = np.load("data/eeg-denoise-net/EMG_all_epochs.npy")

# Normalize
r = 100 / np.sqrt((eeg**2).mean())
eeg *= r
eog *= r
emg *= r

# %% WQN algorithm


def apply_wqn(signal, signal_artifacted):
    cs_signal = pywt.wavedec(signal, "sym5")
    cs_artifact = pywt.wavedec(signal_artifacted, "sym5")

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s)
        ref_order = np.argsort(vals_ref)
        ref_sp = np.linspace(0, len(inv_order) - 1, len(ref_order))
        vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

        r = vals_norm / np.abs(cs)
        cs[:] *= np.minimum(1, r)

    cs_reconstruction = coeffs

    rec = pywt.waverec(coeffs, "sym5")

    return cs_signal, cs_artifact, cs_reconstruction, rec


# %%

data = [
    {
        "signal": eeg[467],
        "artifact": eog[260],
        "title": "EOG",
        "panel": "A",
    },
    {"signal": eeg[438], "artifact": emg[218] / 2, "title": "EMG", "panel": "B"},
]

# %%

fig = plt.figure(figsize=(7.25, 7.25), constrained_layout=True)
# fig.get_layout_engine().set(w_pad=0, h_pad=0, wspace=0, hspace=0)
# fig.get_layout_engine().set(w_pad=20, h_pad=8)

figs = fig.subfigures(
    ncols=2,
    nrows=2,
    width_ratios=(1, 1),
    height_ratios=(2.6, 6),
    hspace=0.01,
    wspace=0.02
)

for i, sample in enumerate(data):
    signal = sample["signal"]
    signal_artifacted = signal + sample["artifact"]
    cs_signal, cs_artifact, cs_reconstruction, rec = apply_wqn(
        signal, signal_artifacted
    )

    subfig = figs[0, i]
    subfig.suptitle(sample["panel"] + "1  " + sample["title"], ha="left", x=0.01)
    ax = subfig.subplots()

    ts = np.arange(signal.size) / 256
    ax.plot(ts, signal_artifacted, lw=1, c=RED, label="Artifacted")
    ax.plot(ts, signal, c="k", lw=1, label="Reference")
    ax.plot(ts, rec, c=BLUE, lw=1, label="WQN")
    ax.set_ylim(-600, 600)
    ax.set_yticks(
        [-350, 0, 350],
        labels=["-350 µV", "0", "+350 µV"],
        va="center",
        rotation=90,
    )
    ax.set_xlabel("Time (s)")

    if i == 0:
        ax.legend()

    subfig = figs[1, i]
    subfig.set_edgecolor("#bbb")
    subfig.set_facecolor("#eee")
    subfig.set_frameon(True)
    subfig.set_linewidth(0.5)
    subfig.suptitle(sample["panel"] + "2", ha="left", x=0.01)

    num_levels = len(cs_signal) - 1
    level_labels = [f"Level $c^\mathrm{{app}}_{num_levels}$"] + [
        f"Level $c_{n}$" for n in range(num_levels, 0, -1)
    ]

    axes = subfig.subplots(
        nrows=num_levels + 1,
        ncols=2,
        gridspec_kw=dict(wspace=0, hspace=0),
    )
    for ax in np.ravel(axes):
        ax.set_facecolor("white")

    for n, (cs_ref, cs_art, cs_rec, level_label) in enumerate(
        zip(cs_signal, cs_artifact, cs_reconstruction, level_labels)
    ):
        cs_ref_abs = np.abs(cs_ref)
        cs_art_abs = np.abs(cs_art)
        cs_rec_abs = np.abs(cs_rec)

        p = np.linspace(0, 1, cs_ref_abs.size)

        axes[n, 0].plot(sorted(cs_ref_abs), p, label="Reference", c="k")
        axes[n, 0].plot(sorted(cs_art_abs), p, label="Artifacted", c=RED)
        axes[n, 0].plot(sorted(cs_rec_abs), p, label="WQN", c=BLUE)
        axes[n, 0].set_ylabel(level_label)

        cs_proj_rec = [np.zeros_like(cs_) for cs_ in cs_reconstruction]
        cs_proj_art = [np.zeros_like(cs_) for cs_ in cs_reconstruction]
        cs_proj_ref = [np.zeros_like(cs_) for cs_ in cs_reconstruction]

        cs_proj_rec[n][:] = cs_rec
        cs_proj_art[n][:] = cs_art
        cs_proj_ref[n][:] = cs_ref

        proj_rec = pywt.waverec(cs_proj_rec, "sym5")
        proj_ref = pywt.waverec(cs_proj_ref, "sym5")
        proj_art = pywt.waverec(cs_proj_art, "sym5")

        axes[n, 1].plot(ts, proj_ref, label="Reference", c="k", lw=1)
        axes[n, 1].plot(ts, proj_art, label="Artifacted", c=RED, lw=1)
        axes[n, 1].plot(ts, proj_rec, label="WQN", c=BLUE)

        axes[n, 0].sharex(axes[0, 0])
        axes[n, 1].sharex(axes[0, 1])
        axes[n, 0].xaxis.set_visible(False)
        axes[n, 1].xaxis.set_visible(False)

        axes[n, 1].tick_params(axis="y", rotation=0)
        axes[n, 1].set_yticks(
            [-200, 0, 200],
            labels=["-200 µV", "0", "200 µV"],
            fontsize=8,
        )
        axes[n, 1].set_ylim(-300, 300)

    axes[0, 0].set_title("Empirical CDF")
    axes[0, 1].set_title("Wavelet projection")
    axes[-1, 0].xaxis.set_visible(True)
    axes[-1, 1].xaxis.set_visible(True)
    axes[-1, 0].set_xlabel("Amplitude")
    axes[-1, 1].set_xlabel("Time (s)")


fig.get_layout_engine().execute(fig)

fig.savefig("output/fig_wqn_mapping.pdf", pad_inches=0.05)
