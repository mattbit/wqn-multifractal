import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.io
import scipy.signal as ss
import wfdb

plt.style.use("./publication.mplstyle")


class WMFA:
    def __init__(self, signal, wavelet="db3"):
        self.signal = signal
        self.wavelet = wavelet
        self._calc_coefficients()

    def _calc_coefficients(self):
        padded_signal = np.pad(self.signal, (1, 1), constant_values=np.inf)
        _, *cs = pywt.wavedec(padded_signal, wavelet=self.wavelet)
        j_max = len(cs)
        scales = np.arange(j_max, 0, -1)

        # Normalize coefficients based on L1 norm
        cs = [c_[np.isfinite(c_)] / (2 ** (0.5 * j_)) for c_, j_ in zip(cs, scales)]

        self.coeffs_ = cs
        self.scales_ = np.arange(len(cs), 0, -1)

    def structure_function(self, q):
        return np.array([(np.abs(d_) ** q).mean() for d_ in self.coeffs_])

    def sup_coeffs(self):
        return np.array([np.abs(d_).max() for d_ in self.coeffs_])

    def inf_coeffs(self):
        return np.array([np.abs(d_).min() for d_ in self.coeffs_])

    def scales_to_freqs(self, sampling_frequency, scales=None):
        if scales is None:
            scales = self.scales_
        return pywt.scale2frequency(self.wavelet, 2**scales) * sampling_frequency


# %% Compare signals.

alpha_band_freq = (4, 16)

# EEG during motor task
raw = scipy.io.loadmat(
    "data/BCICIV/BCICIV_calib_ds1a_1000Hz.mat", struct_as_record=True, simplify_cells=True
)

signal_task = raw["cnt"][:, 5] * 0.1  # µV
freq_task = raw["nfo"]["fs"]
b, a = ss.iirnotch(50, 200, fs=freq_task)
signal_task = ss.filtfilt(b, a, signal_task)

# EEG at rest and with artifact
with np.load("data/physiobank-motion-artifacts-npz/eeg_4.npz") as f:
    signal_rest = f["eeg_ref"] * 1e3  # µV
    signal_rart = f["eeg_art"] * 1e3
    freq_rest = freq_rart = f["freq"]

    b, a = ss.iirnotch(50, 200, fs=freq_rest)
    signal_rest = ss.filtfilt(b, a, signal_rest)
    signal_rart = ss.filtfilt(b, a, signal_rart)


# EEG at sleep
record = wfdb.rdrecord("data/physiobank-psg/0000", smooth_frames=False)
freq_sleep = record.samps_per_frame[2] * record.fs
signal_sleep = record.e_p_signal[2]
signal_sleep = signal_sleep[8000 * 125 : 12000 * 125]
b, a = ss.iirnotch(60, 200, fs=freq_sleep)
signal_sleep = ss.filtfilt(b, a, signal_sleep)


# % Plot

data_props = [
    {
        "title": "EEG during task",
        "signal": signal_task,
        "freq": freq_task,
        "mfa": WMFA(signal_task),
        "view_interval": (20, 25),
    },
    {
        "title": "EEG during rest",
        "signal": signal_rest,
        "freq": freq_rest,
        "mfa": WMFA(signal_rest),
        "view_interval": (82, 87),
        "annotate_alpha_band": True,
    },
    {
        "title": "EEG during sleep",
        "signal": signal_sleep,
        "freq": freq_sleep,
        "mfa": WMFA(signal_sleep),
        "view_interval": (80, 85),
        "psd_interval": (0.5, 70),
        # "annotate_alpha_band": True,
    },
]


fig, axs = plt.subplots(ncols=3, nrows=4, layout="constrained")

# Plot 30 s of signal
for ax, props, n in zip(axs[0], data_props, range(1, 4)):
    title = props["title"]
    signal = props["signal"]
    freq = props["freq"]
    t_min, t_max = props["view_interval"]

    ts = np.arange(signal.size) / freq
    sl = slice((ts > t_min).argmax(), (ts > t_max).argmax())
    ax.set_title(f"A{n}  {title}")
    ax.plot(ts[sl] - ts[sl.start], signal[sl] - signal[sl].mean(), c="k", lw=0.5)

axs[0, 0].set_ylabel("Amplitude")
for ax in axs[0]:
    ax.set_ylim(-75, 75)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks(
        [-50, 0, 50], labels=["-50 µV", "0", "+50 µV"], va="center", rotation=90
    )
    ax.set_xlabel("Time (s)")

# Plot spectrum
import scipy.signal as ss

for ax, props, n in zip(axs[1], data_props, range(1, 4)):
    signal = props["signal"]
    freq = props["freq"]

    fs, S = ss.welch(signal, fs=freq, nperseg=int(freq * 30))

    min_f, max_f = props.get("psd_interval", (0.05, 100))
    mask = (fs <= max_f) & (fs >= min_f)

    ax.set_title(f"B{n}")
    ax.plot(fs[mask], S[mask])

    mask_fit = (fs <= 9) & (fs >= 0.15)

    (a, b), err = np.linalg.lstsq(
        np.vstack([np.log2(fs[mask_fit]), np.ones(mask_fit.sum())]).T,
        np.log2(S[mask_fit]),
        rcond=None,
    )[:2]

    if err < 1e2:
        ax.plot(
            fs[mask_fit],
            2 ** (a * np.log2(fs[mask_fit]) + b),
            c="navy",
            ls="--",
            lw=0.5,
        )
        ax.text(
            10**np.log10(fs[mask_fit]).mean(),
            2 ** (a * np.log2(fs[mask_fit]) + b).mean(),
            rf"$\alpha = {-a:.2f}$",
            ha="center",
            va="bottom",
            rotation=np.rad2deg(np.arctan(a)) - 22,
            transform_rotates_text=True,
            fontsize=8,
        )

    if props.get("annotate_alpha_band", False):
        # centering done by hand…
        # ax.vlines(alpha_band_freq, 0, 2, color="r", lw=0.25)

        band_x_center = 0.5 * np.sum(alpha_band_freq) - 2
        band_w = alpha_band_freq[1] - alpha_band_freq[0]

        yh = S[(fs <= alpha_band_freq[1]) & (fs >= alpha_band_freq[0])].min()

        ax.annotate(
            "α-band",
            xy=(band_x_center, 0.1),
            xycoords=("data", "data"),
            textcoords=("data", "axes fraction"),
            xytext=(band_x_center, 0.2),
            fontsize=9,
            ha="center",
            va="top",
            arrowprops=dict(
                arrowstyle=f"-[, widthB={band_w/10}, lengthB=0.5", lw=1, color="navy"
            ),
        )


axs[1, 0].set_ylabel("PSD")
for ax in axs[1]:
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(10.0 ** np.arange(-1, 3))
    ax.set_xlabel("Frequency (Hz)")


# Plot structure functions

mprops = dict(markeredgewidth=1, marker=".")

for ax, props, n in zip(axs[2], data_props, range(1, 4)):
    mfa = props["mfa"]
    freq = props["freq"]
    mask = props.get("struct_fun_mask", np.ones(mfa.scales_.size, dtype=bool))

    fs = mfa.scales_to_freqs(freq)
    fs_mask = (fs >= 0.1) & (fs <= 49)

    alpha_band_scale = (
        mfa.scales_[(fs >= alpha_band_freq[0]).argmax()],
        mfa.scales_[(fs >= alpha_band_freq[1]).argmax()],
    )

    S_q1 = mfa.structure_function(q=1)
    S_q2 = mfa.structure_function(q=2)

    jj = mfa.scales_[fs_mask]
    (a1, b1), err1 = np.linalg.lstsq(
        np.vstack([jj, np.ones_like(jj)]).T, np.log2(S_q1[fs_mask]), rcond=None
    )[:2]
    (a2, b2), err2 = np.linalg.lstsq(
        np.vstack([jj, np.ones_like(jj)]).T, np.log2(S_q2[fs_mask]), rcond=None
    )[:2]

    (l,) = ax.plot(mfa.scales_, np.log2(S_q1), **mprops, lw=0.5, label="q = 1")
    # ax.plot(mfa.scales_, a1 * mfa.scales_ + b1, c="navy", lw=0.5, ls="--")
    if err1 < 5:
        ax.plot(
            mfa.scales_[fs_mask],
            a1 * mfa.scales_[fs_mask] + b1,
            c="navy",
            ls="--",
            lw=0.5,
        )
        ax.text(
            mfa.scales_[fs_mask].mean(),
            (a1 * mfa.scales_[fs_mask] + b1).mean() - 0.5,
            rf"$\eta = {a1:.2f}$",
            ha="center",
            va="bottom",
            rotation=np.rad2deg(np.arctan(a1)),
            transform_rotates_text=True,
            fontsize=8,
        )
    ax.set_title(f"C{n}")
    ax.plot(
        mfa.scales_,
        np.log2(S_q2),
        **mprops,
        fillstyle="none",
        lw=0.5,
        label="q = 2",
    )

    if err2 < 5:
        ax.plot(
            mfa.scales_[fs_mask],
            a2 * mfa.scales_[fs_mask] + b2,
            c="navy",
            ls="--",
            lw=0.5,
        )
        ax.text(
            mfa.scales_[fs_mask].mean(),
            (a2 * mfa.scales_[fs_mask] + b2).mean() - 1,
            rf"$\eta = {a2:.2f}$",
            ha="center",
            va="bottom",
            rotation=np.rad2deg(np.arctan(a2)),
            transform_rotates_text=True,
            fontsize=8,
        )

    if props.get("annotate_alpha_band", False):
        band_x_center = 0.5 * np.sum(alpha_band_scale)
        band_w = (alpha_band_scale[0] - alpha_band_scale[1]) / 2
        yh = np.log2(S_q2[(band_x_center >= mfa.scales_).argmax()])

        ax.annotate(
            "α-band",
            xy=(band_x_center, yh + 1),
            xycoords=("data", "data"),
            textcoords=("data", "axes fraction"),
            xytext=(band_x_center, 0.8),
            fontsize=9,
            ha="center",
            va="bottom",
            arrowprops=dict(
                arrowstyle=f"-[, widthB={band_w/2}, lengthB=0.5", lw=1, color="navy"
            ),
        )

    # Interval 0.1–50 Hz
    ax.axvspan(
        mfa.scales_[fs_mask].min() - 0.5,
        mfa.scales_[fs_mask].max() + 0.5,
        fc=(0, 0, 0.5, 0.05),
        zorder=-1,
    )

axs[2, 0].set_ylabel("$\\log_2 S_f(j, q)$")
axs[2, 0].legend()
for ax in axs[2]:
    ax.set_xlabel("Wavelet scale $j$")

# H_min

for ax, props, n in zip(axs[3], data_props, range(1, 4)):
    mfa = props["mfa"]
    freq = props["freq"]

    fs = mfa.scales_to_freqs(freq)
    fs_mask = (fs >= 0.1) & (fs <= 49)
    jj = mfa.scales_[fs_mask]

    (h_min, b), err = np.linalg.lstsq(
        np.vstack([jj, np.ones_like(jj)]).T,
        np.log2(mfa.sup_coeffs()[fs_mask]),
        rcond=None,
    )[:2]

    ax.set_title(f"D{n}")
    ax.plot(mfa.scales_, np.log2(mfa.sup_coeffs()), marker=".")

    if err < 15:
        ax.plot(
            mfa.scales_[fs_mask],
            h_min * mfa.scales_[fs_mask] + b,
            c="navy",
            ls="--",
            lw=0.5,
        )
        ax.text(
            mfa.scales_[fs_mask].mean(),
            (h_min * mfa.scales_[fs_mask] + b).mean(),
            rf"$H^\mathrm{{min}} = {h_min:.2f}$",
            ha="center",
            va="top",
            rotation=np.rad2deg(np.arctan(h_min)),
            transform_rotates_text=True,
        )

    # Interval 0.1–50 Hz
    ax.axvspan(
        mfa.scales_[fs_mask].min() - 0.5,
        mfa.scales_[fs_mask].max() + 0.5,
        fc=(0, 0, 0.5, 0.05),
        zorder=-1,
    )

axs[3, 0].set_ylabel("$\\log_2 \\sup \\left|c_{j,k}\\right|$")
for ax in axs[3]:
    ax.set_xlabel("Wavelet scale $j$")

# Final retouching
axs[1, 2].set_ylim(11**-3, 9.9**3)

fig.set_constrained_layout_pads(h_pad=0)
fig.set_size_inches(7.25, 7.25)
fig.savefig("output/fig_scale_invariance.pdf")

