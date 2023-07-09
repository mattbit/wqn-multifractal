"""Converts the PhysioBank formatted data into numpy arrays."""

from pathlib import Path

import numpy as np
import wfdb

dataset_path = Path("data/physiobank-motion-artifacts")
output_path = Path("data")
output_path.joinpath("physiobank-motion-artifacts-npz").mkdir(parents=True, exist_ok=True)

for path in dataset_path.glob("*.hea"):
    record_name = path.stem
    record_path = str(path.with_name(record_name))
    record = wfdb.rdrecord(record_path)
    annots = wfdb.rdann(record_path, "trigger")

    fs = record.fs
    eeg = record.p_signal[:, :2]

    reference = eeg[:, 0]
    signal = eeg[:, 1]

    np.savez(
        output_path.joinpath("physiobank-motion-artifacts-npz", record_name + ".npz"),
        eeg_art=signal,
        eeg_ref=reference,
        freq=fs,
    )

    # Generate Matlab files if needed
    # scipy.io.savemat(
    #     output_path.joinpath("physiobank-motion-artifacts-mat", record_name + ".mat"),
    #     {"eeg_art": signal, "eeg_ref": reference, "freq": fs},
    # )
