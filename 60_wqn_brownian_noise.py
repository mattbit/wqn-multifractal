import matplotlib.pyplot as plt
import numpy as np
import pywt

plt.style.use("./publication.mplstyle")

RED = "#fc8d62"
GREEN = "#66c2a5"
GREEN_L = "#7ad6b9"
BLUE = "#8da0cb"
BLUE_L = "#a1b4df"

# %% WQN algorithm

_wavelet = "db5"
_mode = "periodic"


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

fig = plt.figure(figsize=(7.25, 3.5), constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.02)

figA, figB = fig.subfigures(ncols=2, width_ratios=(1, 1))

xspace = np.linspace(0, 8 * np.pi, LENGTH)
ref_signal = np.sin(xspace)

σs = [1, 2, 5, 10]
axs = figA.subplots(nrows=len(σs), sharex=True)

for n, σ in enumerate(σs):
        
    artifact = np.random.normal(scale=σ, size=LENGTH)
    art_signal = ref_signal + artifact
    rec_signal = remove_artifact(
        ref_signal, art_signal
    )
    if n == 0:
        axs[n].plot(xspace, art_signal, color=RED, label="Artifacted")
    axs[n].plot(xspace, ref_signal, color="k", ls="--", lw=0.5, label="Original")
    axs[n].plot(xspace, rec_signal, color=BLUE, label="WQN")
    axs[n].set_title(f"$\sigma$ = {σ}", loc="center", pad=0)

    if n == 1:
        axs[n].legend(ncol=2, loc="lower left")

figA.suptitle("A", fontweight="bold", x=0.09, y=0.99)        

# Figure B
σs = np.linspace(0.1, 50, 1000)

art_mses = []
rec_mses = []
for σ in σs:
    artifact = np.random.normal(scale=σ, size=LENGTH)
    art_signal = ref_signal + artifact
    rec_signal = remove_artifact(
        ref_signal, art_signal
    )
    rec_mses.append(mse(ref_signal, rec_signal))
    art_mses.append(mse(ref_signal, art_signal))


figB.suptitle("B", fontweight="bold", x=0.09, y=0.99)
ax = figB.subplots()
ax.set_title(" ")

ax.plot(σs, art_mses, color=RED, label="Artifacted")
ax.plot(σs, rec_mses, color=BLUE, label="WQN corrected")
ax.legend()

ax.set_yscale("log")
ax.set_ylabel("Mean Squared Error (MSE)")
ax.set_xlabel("Noise amplitude σ")

fig.savefig("output/fig_brownian_noise.pdf", pad_inches=0.05)
