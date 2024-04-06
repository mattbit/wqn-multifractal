import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import get_window

plt.style.use("./publication.mplstyle")

RED = "#fc8d62"
GREEN = "#66c2a5"
GREEN_L = "#7ad6b9"
BLUE = "#8da0cb"
BLUE_L = "#a1b4df"

# %% WQN algorithm

_wavelet = "db5"
_mode = "periodization"


def remove_artifact(reference, artifact, level=None, with_coeffs=False):
    cs_signal = pywt.wavedec(reference, _wavelet, level=level, mode=_mode)
    cs_artifact = pywt.wavedec(artifact, _wavelet, level=level, mode=_mode)

    coeffs = [c.copy() for c in cs_artifact]
    for cs_s, cs in zip(cs_signal, coeffs):
        order = np.argsort(np.abs(cs))
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        vals_ref = np.abs(cs_s)
        ref_order = np.argsort(vals_ref)
        ref_sp = np.linspace(0, len(inv_order), len(ref_order))
        vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

        r = vals_norm / np.abs(cs).clip(1e-30)
        cs[:] *= np.minimum(1, r)

    rec = pywt.waverec(coeffs, _wavelet, mode=_mode)

    if with_coeffs:
        return rec, coeffs

    return rec


def mse(a, b):
    return ((a - b) ** 2).mean()


# %%

np.random.seed(1239)
LENGTH = 10_000

fig = plt.figure(figsize=(7.25, 4.6), constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.01, wspace=0.02)

figs = fig.subfigures(ncols=2, nrows=2, width_ratios=(1, 1), height_ratios=(4, 2))

figA1 = figs[0, 0]
figA2 = figs[1, 0]
figB1 = figs[0, 1]
figB2 = figs[1, 1]

figA1.suptitle("A1", fontweight="bold", x=0.095, y=1)
figA2.suptitle("A2", fontweight="bold", x=0.095, y=0.98)
figB1.suptitle("B1", fontweight="bold", x=0.095, y=1)
figB2.suptitle("B2", fontweight="bold", x=0.095, y=0.98)

xspace = np.linspace(0, 8 * np.pi, LENGTH)
ref_signal = np.sin(xspace)

σs = [1, 2, 5, 10]
axs1 = figA1.subplots(nrows=len(σs), sharex=True)
axs2 = figB1.subplots(nrows=len(σs), sharex=True)

w = get_window(("kaiser", 64), ref_signal.size)

for n, σ in enumerate(σs):

    artifact = np.random.normal(scale=σ, size=LENGTH)

    art_signal = ref_signal + artifact
    rec_signal = remove_artifact(ref_signal, art_signal)

    art_signal2 = ref_signal + artifact * w
    rec_signal2 = remove_artifact(ref_signal, art_signal2)

    if n == 0:
        axs1[n].plot(xspace, art_signal, color=RED, label="Artifacted")

    axs1[n].plot(xspace, ref_signal, color="k", ls="--", lw=0.5, label="Original")
    axs1[n].plot(xspace, rec_signal, color=BLUE, label="WQN")
    axs1[n].set_title(f"$\sigma$ = {σ}", loc="center", pad=0)

    axs2[n].plot(xspace, ref_signal, color="k", ls="--", lw=0.5, label="Original")
    axs2[n].plot(xspace, rec_signal2, color=BLUE, label="WQN")
    axs2[n].set_title(f"$\sigma$ = {σ}", loc="center", pad=0)

    if n == 0:
        axs2[n].plot(xspace, art_signal2, color=RED, label="Artifacted", zorder=-1)
        axs2[n].legend(ncol=2, loc="lower left", mode="expand", labelspacing=0.5)


# Figure B
σs = np.linspace(0.1, 50, 1000)

art_mses = []
art2_mses = []
rec_mses = []
rec2_mses = []
for σ in σs:
    artifact = np.random.normal(scale=σ, size=LENGTH)

    art_signal = ref_signal + artifact
    rec_signal = remove_artifact(ref_signal, art_signal)

    art_signal2 = ref_signal + artifact * w
    rec_signal2 = remove_artifact(ref_signal, art_signal2)

    rec_mses.append(mse(ref_signal, rec_signal))
    art_mses.append(mse(ref_signal, art_signal))

    art2_mses.append(mse(ref_signal, art_signal2))
    rec2_mses.append(mse(ref_signal, rec_signal2))

ax = figA2.subplots()

ax.plot(σs, art_mses, color=RED, label="Artifacted")
ax.plot(σs, rec_mses, color=BLUE, label="WQN corrected")
ax.legend(loc="upper right")

ax.set_yscale("log")
ax.set_ylabel("MSE")
ax.set_xlabel("Noise amplitude σ")
ax.set_ylim(1e-3, 1e4)
ax.set_title(" ")

ax = figB2.subplots()

ax.plot(σs, art2_mses, color=RED, label="Artifacted")
ax.plot(σs, rec2_mses, color=BLUE, label="WQN corrected")

ax.set_yscale("log")
ax.set_xlabel("Noise amplitude σ")
ax.set_ylim(1e-3, 1e4)
ax.set_title(" ")

# Retouch
axs1[0].set_ylim(-6, 6)
axs2[0].set_ylim(-6, 6)

fig.savefig("output/fig_brownian_noise.pdf", pad_inches=0.05)
