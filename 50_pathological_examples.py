import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./publication.mplstyle")

RED = "#fc8d62"
GREEN = "#66c2a5"
GREEN_L = "#7ad6b9"
BLUE = "#8da0cb"
BLUE_L = "#a1b4df"

# %%

def map_coeffs(cs, cs_s):
    order = np.argsort(np.abs(cs))
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    vals_ref = np.abs(cs_s)
    ref_order = np.argsort(vals_ref)
    ref_sp = np.linspace(0, len(inv_order) - 1, len(ref_order))
    vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])
    r = np.nan_to_num(vals_norm / np.abs(cs))
    return cs * np.minimum(1, r)


def ht(x, t):
    """Hard thresholding."""
    return np.where(np.abs(x) <= t, x, 0)


def st(x, t):
    """Soft thresholding."""
    return x.clip(-t, t)


def cdf(x):
    return np.linspace(0, 1, len(x)), x[np.argsort(x)]


# %%

LENGTH = 2000
AMPLITUDE = 1
ART_WIDTH = LENGTH // 6

square_artifact = np.zeros(LENGTH)
square_artifact[LENGTH // 2 - ART_WIDTH : LENGTH // 2 + ART_WIDTH] = AMPLITUDE

triangle_artifact = np.zeros(LENGTH)
triangle_artifact[LENGTH // 2 - ART_WIDTH : LENGTH // 2] = np.linspace(
    0, AMPLITUDE, ART_WIDTH
)
triangle_artifact[LENGTH // 2 : LENGTH // 2 + ART_WIDTH] = np.linspace(
    AMPLITUDE, 0, ART_WIDTH
)

cosine_artifact = np.zeros(LENGTH)
_cos_wave = 0.5 * (np.cos(np.linspace(-np.pi, np.pi, 2 * ART_WIDTH)) + 1)
cosine_artifact[LENGTH // 2 - ART_WIDTH : LENGTH // 2 + ART_WIDTH] = (
    AMPLITUDE * _cos_wave
)

# %

ref_signal = np.linspace(0, 1, LENGTH)

data_props = [
    {
        "index": "A",
        "name": "Square wave",
        "artifact": square_artifact,
    },
    {
        "index": "B",
        "name": "Triangle wave",
        "artifact": triangle_artifact,
    },
    {
        "index": "C",
        "name": "Smooth artifact",
        "artifact": cosine_artifact,
    },
]

xspace = np.linspace(0, 1, LENGTH)

main_fig = plt.figure(figsize=(7.25, 6.25), constrained_layout=True)
figs = main_fig.subfigures(nrows=3)

for n_fig, data in enumerate(data_props):
    fig = figs[n_fig]
    axs = fig.subplots(ncols=3, sharey=True)

    fig.supylabel(data["name"], fontweight="medium")

    art_signal = ref_signal + data["artifact"]

    axs[0].plot(xspace, ref_signal, zorder=10, label="Reference")
    axs[0].plot(xspace, art_signal, label="Artifacted")

    axs[1].plot(*cdf(ref_signal), zorder=10, label="Reference")
    axs[1].plot(*cdf(art_signal), label="Artifacted")

    wqn_coeffs = map_coeffs(art_signal, ref_signal)
    threshold = ref_signal.max()
    ht_coeffs = ht(art_signal, threshold)
    st_coeffs = st(art_signal, threshold)

    mse_wqn = np.mean((wqn_coeffs - ref_signal) ** 2)
    mse_ht = np.mean((ht_coeffs - ref_signal) ** 2)
    mse_st = np.mean((st_coeffs - ref_signal) ** 2)

    axs[2].plot(xspace, ref_signal, zorder=10, label="Reference")
    axs[2].plot(xspace, ht_coeffs, ls="--", label=f"HT (MSE = {mse_ht:.2f})", c="gray")
    axs[2].plot(xspace, st_coeffs, ls=":", label=f"ST (MSE = {mse_st:.2f})", c="gray")
    axs[2].plot(xspace, wqn_coeffs, zorder=5, label=f"WQN (MSE = {mse_wqn:.2f})", c=BLUE)
    axs[2].legend()

    axs[0].set_ylim(-0.1, 2.1)
    axs[0].set_yticks([0, 1, 2])

    if n_fig == 0:
        axs[0].set_title("Wavelet coefficients", loc="center")
        axs[1].set_title("Quantile function", loc="center")
        axs[2].set_title("Reconstructed coefficients", loc="center")
        axs[0].legend(loc="lower right")
        axs[1].legend(loc="lower right")

    if n_fig == 2:
        axs[0].set_xlabel("Time")
        axs[-1].set_xlabel("Time")
        axs[1].set_xlabel("Probability")

    fig.suptitle(data["index"], x=0.01, y=0.98, fontweight="bold")
    
main_fig.savefig("output/fig_pathological_examples.pdf", pad_inches=0.05)
