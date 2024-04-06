import numpy as np
import scipy.signal as ss
import pywt


def calc_rmse(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).mean())


def filter_bandpass(signal, low, high, fs, order=2):
    Wn = 2 * np.array([low, high]) / fs
    sos = ss.butter(order, Wn, btype="bandpass", output="sos")

    return ss.sosfiltfilt(sos, signal, axis=0)


def filter_highpass(signal, freq, fs, order=2):
    w = 2 * freq / fs
    sos = ss.butter(order, w, btype="highpass", output="sos")

    return ss.sosfiltfilt(sos, signal, axis=0)


def mask_to_intervals(mask, index=None):
    """Convert a boolean mask to a sequence of intervals.
    Caveat: when no index is given, the returned values correspond to the
    Python pure integer indexing (starting element included, ending element
    excluded). When an index is passed, pandas label indexing convention
    with strict inclusion is used.
    For example `mask_to_intervals([0, 1, 1, 0])` will return `[(1, 3)]`,
    but `mask_to_intervals([0, 1, 1, 0], ["a", "b", "c", "d"])` will return
    the value `[("b", "c")]`.
    Parameters
    ----------
    mask : numpy.ndarray
        A boolean array.
    index : Sequence, optional
        Elements to use as indices for determining interval start and end. If
        no index is given, integer array indices are used.
    Returns
    -------
    intervals : Sequence[Tuple[Any, Any]]
        A sequence of (start_index, end_index) tuples. Mindful of the caveat
        described above concerning the indexing convention.
    """
    if not np.any(mask):
        return []

    edges = np.flatnonzero(np.diff(np.pad(mask, 1)))
    intervals = edges.reshape((len(edges) // 2, 2))

    if index is not None:
        return [(index[i], index[j - 1]) for i, j in intervals]

    return [(i, j) for i, j in intervals]


def intervals_to_mask(intervals, size=None):
    mask = np.zeros(size, dtype=bool)
    for i, j in intervals:
        mask[i:j] = True

    return mask


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
