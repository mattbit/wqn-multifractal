from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scipy.signal as ss
from tqdm import tqdm

import tools

plt.style.use("./publication.mplstyle")

RED = "#fc8d62"
GREEN = "#66c2a5"
GREEN_L = "#7ad6b9"
BLUE = "#8da0cb"
BLUE_L = "#a1b4df"

# %% WQN algorithm


def apply_wqn(
    signal, signal_artifacted, wavelet="sym5", mode="antisymmetric", max_level=None
):
    cs_signal = pywt.wavedec(signal, wavelet, level=max_level, mode=mode)
    cs_artifact = pywt.wavedec(signal_artifacted, wavelet, level=max_level, mode=mode)

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s)
        ref_order = np.argsort(vals_ref)
        ref_sp = np.linspace(0, len(inv_order) - 1, len(ref_order))
        vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

        r = np.nan_to_num(vals_norm / np.abs(cs))
        cs[:] *= np.minimum(1, r)

    cs_reconstruction = coeffs

    rec = pywt.waverec(coeffs, wavelet, mode)

    return cs_signal, cs_artifact, cs_reconstruction, rec


# %% Load data

f = np.load("data/physiobank-motion-artifacts-npz/eeg_20.npz")
freq = f["freq"]

sel = slice(100_000, 200_000)

signal = 1e3 * tools.filter_highpass(f["eeg_ref"], 0.2, freq)[sel]
artifacted = 1e3 * tools.filter_highpass(f["eeg_art"], 0.2, freq)[sel]

wavelets = [
    {"title": "Symlet 5", "wavelet": "sym5"},
    {"title": "Daubechies 5", "wavelet": "db5"},
    {"title": "Coiflet 3", "wavelet": "coif3"},
    {"title": "Biorthogonal spline 3.5", "wavelet": "bior3.5"},
    {"title": "Discrete Meyer", "wavelet": "dmey"},
]


fig, axs = plt.subplots(
    ncols=2,
    nrows=(len(wavelets) + 1) // 2,
    figsize=(7.25, 3.2),
    sharey=True,
    sharex=True,
    constrained_layout=True,
)
faxs = np.ravel(axs)
art_rmse = tools.calc_rmse(signal, artifacted)

ts = np.arange(len(signal)) / freq
(la,) = faxs[0].plot(ts, artifacted, c=RED, lw=0.5, label="Artifacted")
(lo,) = faxs[0].plot(ts, signal, c="k", lw=0.5, label="EEG")
faxs[0].set_title("Artifacted EEG")
axs[0, 0].legend(handles=[lo, la], loc="lower left", ncol=2, mode="expand")

axs[0, 0].text(
    0.05,
    0.8,
    f"Artifact RMSE: {art_rmse:.1f} µV",
    transform=axs[0, 0].transAxes,
)

for n, wdata in enumerate(wavelets):
    *_, rec = apply_wqn(
        signal, artifacted, wavelet=wdata["wavelet"], mode="antisymmetric"
    )
    rmse = tools.calc_rmse(signal, rec)
    faxs[n + 1].plot(ts, signal, c="k", lw=0.5)
    faxs[n + 1].plot(ts, rec, lw=0.5, label="WQN")
    faxs[n + 1].set_title(f"{wdata['title']}")
    faxs[n + 1].text(
        0.05,
        0.8,
        f"WQN RMSE: {rmse:.2f} µV",
        transform=faxs[n + 1].transAxes,
    )


axs[0, 1].legend()

axs[-1, 0].set_xlabel("Time (s)")
axs[-1, 1].set_xlabel("Time (s)")
for ax in axs[:, 0]:
    ax.set_yticks([-50, 0, 50], labels=["-50 µV", "0", "50 µV"])

fig.savefig("output/fig_basis_independence.pdf")

# %% Systematic results

all_epochs_art = []
all_epochs_sig = []

for i in range(1, 24):
    f = np.load(f"data/physiobank-motion-artifacts-npz/eeg_{i}.npz")
    freq = f["freq"]

    signal = 1e3 * tools.filter_highpass(f["eeg_ref"], 0.1, freq)
    artifacted = 1e3 * tools.filter_highpass(f["eeg_art"], 0.1, freq)

    # Split in 20 s epochs
    epoch_len = int(freq * 20)
    signal = signal[: signal.size - signal.size % epoch_len]
    artifacted = artifacted[: signal.size - signal.size % epoch_len]

    window = ss.get_window(("kaiser", 5), epoch_len)
    epochs_sig = np.array(np.split(signal, signal.size // epoch_len))
    epochs_art = (
        epochs_sig
        + np.array(np.split(artifacted - signal, signal.size // epoch_len)) * window
    )

    # Keep only those with significant artifacts
    mask = ((epochs_sig - epochs_art) ** 2).mean(axis=-1) > 100
    epochs_sig = epochs_sig[mask]
    epochs_art = epochs_art[mask]

    all_epochs_sig.extend(epochs_sig)
    all_epochs_art.extend(epochs_art)

all_epochs_sig = np.asarray(all_epochs_sig)
all_epochs_art = np.asarray(all_epochs_art)

# %%

import pandas as pd

results = []

for wdata in wavelets:
    mse = []
    for es, ea in zip(all_epochs_sig, all_epochs_art):
        art = es - ea
        nart = 2 * art / np.sqrt((art**2).mean())
        nes = es / np.sqrt((es**2).mean())

        *_, er = apply_wqn(
            nes, nes + nart, wavelet=wdata["wavelet"], mode="antisymmetric"
        )
        mse.append(((er - nes) ** 2).mean())

    results.append(
        {
            "wavelet": wdata["wavelet"],
            "title": wdata["title"],
            "mse_avg": np.mean(np.sqrt(mse)),
            "mse_std": np.std(np.sqrt(mse)),
        }
    )


df = pd.DataFrame(results)

# %%

results = []
for wavelet, moments in [
    ("sym2", 2),
    ("sym3", 3),
    ("sym4", 4),
    ("sym5", 5),
    ("sym6", 6),
    ("sym7", 7),
    ("sym8", 8),
    ("coif1", 1),
    ("coif2", 2),
    ("coif3", 3),
    ("coif4", 4),
    ("coif5", 5),
    ("coif6", 6),
    ("coif7", 7),
    ("coif8", 8),
]:
    mse = []
    for es, ea in zip(all_epochs_sig, all_epochs_art):
        *_, er = apply_wqn(es, ea, wavelet=wavelet, mode="antisymmetric")
        mse.append(((er - es) ** 2).mean())

    results.append(
        {
            "wavelet": wavelet,
            "vanishing_moments": moments,
            "mse_avg": np.mean(np.sqrt(mse)),
            "mse_std": np.std(np.sqrt(mse)),
        }
    )


df = pd.DataFrame(results)


# %% Bootstrapping

# Load data and normalize


def bootstrap(shape, examples):
    ref_coeffs = pywt.wavedec(
        examples, "db3", level=None, axis=-1, mode="periodization"
    )
    coeffs = pywt.wavedec(np.zeros(shape), "db3", mode="periodization")
    for cs, r_cs in zip(coeffs, ref_coeffs):
        cs[:] = np.random.choice(r_cs.reshape(-1), size=cs.shape, replace=True)

    return pywt.waverec(coeffs, "db3", mode="periodization")


# %% Simulation

if Path("data/bootstrapped_signals.npz").exists():
    with np.load("data/bootstrapped_signals.npz") as f:
        epochs_sig = f["epochs_sig"]
        epochs_art_eog = f["epochs_art_eog"]
        epochs_art_emg = f["epochs_art_emg"]
else:
    win_fn = ss.get_window(("kaiser", 16), 256 * 300)
    eeg = np.load("data/eeg-denoise-net/EEG_all_epochs.npy")
    eog = np.load("data/eeg-denoise-net/EOG_all_epochs.npy")
    emg = np.load("data/eeg-denoise-net/EMG_all_epochs.npy")

    n_eeg = (eeg - eeg.mean(axis=-1).reshape(-1, 1)) / eeg.std(axis=-1).reshape(-1, 1)
    n_eog = (eog - eog.mean(axis=-1).reshape(-1, 1)) / eog.std(axis=-1).reshape(-1, 1)
    n_emg = (emg - emg.mean(axis=-1).reshape(-1, 1)) / emg.std(axis=-1).reshape(-1, 1)

    eeg_coeffs = pywt.wavedec(n_eeg, "db3", level=None, axis=-1, mode="periodization")
    eog_coeffs = pywt.wavedec(n_eog, "db3", level=None, axis=-1, mode="periodization")
    emg_coeffs = pywt.wavedec(n_emg, "db3", level=None, axis=-1, mode="periodization")

    num_samples = 1024
    epochs_sig = bootstrap((1000, 256 * 300), n_eeg)
    epochs_art_eog = bootstrap((1000, 256 * 300), n_eog) * win_fn
    epochs_art_emg = bootstrap((1000, 256 * 300), n_emg) * win_fn
    np.savez(
        "data/bootstrapped_signals.npz",
        epochs_sig=epochs_sig,
        epochs_art_eog=epochs_art_eog,
        epochs_art_emg=epochs_art_emg,
    )

epochs_sig

# %% Comparison of families

wavelets = [
    {"title": "Symlet 5", "wavelet": "sym5"},
    {"title": "Daubechies 5", "wavelet": "db5"},
    {"title": "Coiflet 3", "wavelet": "coif3"},
    {"title": "Biorthogonal spline 3.5", "wavelet": "bior3.5"},
    {"title": "Discrete Meyer", "wavelet": "dmey"},
]
l = 9

if Path("output/wavelet_basis_comp.csv").exists():
    df = pd.read_csv("output/wavelet_basis_comp.csv")
else:
    results = []
    for wdata in tqdm(wavelets):
        wavelet = wdata["wavelet"]

        mse_eog = []
        for es, ea in zip(epochs_sig, epochs_art_eog):
            *_, er = apply_wqn(
                es, es + ea, wavelet=wavelet, mode="antisymmetric", max_level=l
            )
            mse_eog.append(((er - es) ** 2).mean())

        mse_emg = []
        for es, ea in zip(epochs_sig, epochs_art_emg):
            *_, er = apply_wqn(
                es, es + ea, wavelet=wavelet, mode="antisymmetric", max_level=l
            )
            mse_emg.append(((er - es) ** 2).mean())

        results.append(
            {
                "wavelet": wavelet,
                "eog_mse_avg": np.mean(np.sqrt(mse_eog)),
                "eog_mse_std": np.std(np.sqrt(mse_eog)),
                "emg_mse_avg": np.mean(np.sqrt(mse_emg)),
                "emg_mse_std": np.std(np.sqrt(mse_emg)),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("output/wavelet_basis_comp.csv", index=False)


categories = dict(zip(df.wavelet.values, df.wavelet.index))

fig, (axl, axr) = plt.subplots(
    ncols=2, figsize=(7.25, 2.5), sharey=True, constrained_layout=True
)
for w, dx in df.groupby("wavelet"):
    aeog = axl.bar(
        categories[w] - 0.2, dx.eog_mse_avg, width=0.35, fc=GREEN, label="EOG"
    )
    axl.errorbar(
        categories[w] - 0.2,
        dx.eog_mse_avg,
        dx.eog_mse_std,
        c=GREEN_L,
        fmt="none",
        capsize=3,
        markeredgewidth=1,
        elinewidth=1,
    )
    aemg = axl.bar(
        categories[w] + 0.2, dx.emg_mse_avg, width=0.35, fc=BLUE, label="EMG"
    )
    axl.errorbar(
        categories[w] + 0.2,
        dx.emg_mse_avg,
        dx.emg_mse_std,
        c=BLUE_L,
        fmt="none",
        capsize=3,
        markeredgewidth=1,
        elinewidth=1,
    )

axl.set_xticks(list(categories.values()), labels=categories.keys())
axl.legend(handles=[aeog, aemg])
axl.set_ylabel("RMSE")
axl.set_xlabel("Wavelet")

# % Comparison of vanishing moments

wavelets = [
    "db1",
    "db2",
    "db3",
    "db4",
    "db5",
    "db6",
    "db7",
    "db8",
    "db9",
    "db10",
    "sym2",
    "sym3",
    "sym4",
    "sym5",
    "sym6",
    "sym7",
    "sym8",
    "sym9",
    "sym10",
    "coif1",
    "coif2",
    "coif3",
    "coif4",
    "coif5",
]

if Path("output/wavelet_moments_comp.csv").exists():
    df = pd.read_csv("output/wavelet_moments_comp.csv")
else:
    results = []
    for wavelet in tqdm(wavelets):
        mse = []
        for es, ea in zip(epochs_sig, epochs_art_eog):
            l = pywt.dwt_max_level(es.size, wavelet) - 1
            l = 9
            *_, er = apply_wqn(
                es, es + ea, wavelet=wavelet, mode="antisymmetric", max_level=l
            )
            mse.append(((er - es) ** 2).mean())

        results.append(
            {
                "wavelet": wavelet,
                "mse_avg": np.mean(np.sqrt(mse)),
                "mse_std": np.std(np.sqrt(mse)),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("output/wavelet_moments_comp.csv", index=False)


df["family"] = [pywt.Wavelet(w).short_family_name for w in df.wavelet]
df["vanishing_moments"] = [pywt.Wavelet(w).vanishing_moments_psi for w in df.wavelet]

ax = axr
df_db = df[df.family == "db"]
df_sym = df[df.family == "sym"]
df_coif = df[df.family == "coif"]

cat_space = 0.25
c0 = ax.scatter(
    df_db.vanishing_moments - cat_space, df_db.mse_avg, marker="^", label="Daubechies"
)
ax.errorbar(
    df_db.vanishing_moments - cat_space,
    df_db.mse_avg,
    df_db.mse_std,
    fmt="none",
    capsize=3,
    markeredgewidth=1,
    elinewidth=1,
    c=c0.get_facecolor(),
)

c1 = ax.scatter(df_sym.vanishing_moments, df_sym.mse_avg, marker="o", label="Symlets")
ax.errorbar(
    df_sym.vanishing_moments,
    df_sym.mse_avg,
    df_sym.mse_std,
    fmt="none",
    capsize=3,
    markeredgewidth=1,
    elinewidth=1,
    c=c1.get_facecolor(),
)

c2 = ax.scatter(
    df_coif.vanishing_moments + cat_space, df_coif.mse_avg, marker="s", label="Coiflets"
)
ax.errorbar(
    df_coif.vanishing_moments + cat_space,
    df_coif.mse_avg,
    df_coif.mse_std,
    fmt="none",
    capsize=3,
    markeredgewidth=1,
    elinewidth=1,
    c=c2.get_facecolor(),
)

ax.set_xticks(np.arange(1, 11))
ax.set_ylim(0)

ax.set_xlabel("Number of vanishing moments")
ax.legend()

axl.set_title("A")
axr.set_title("B")

fig.savefig("output/fig_basis_independence_2.pdf")


