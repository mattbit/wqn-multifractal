# Physiobank motion artifacts data

The data is described in these papers:

- Sweeney KT, Ayaz H, Ward TE, Izzetoglu M, McLoone SF, Onaral B. A Methodology for Validating Artifact Removal Techniques for Physiological Signals. IEEE Trans Info Tech Biomed 16(5):918-926; 2012 (Sept).
- Sweeney KT, McLoone SF, Ward TE. The Use of Ensemble Empirical Mode Decomposition With Canonical Correlation Analysis as a Novel Artifact Removal Technique. IEEE Trans Biomed Eng 60(1):97-105; 2013 (Jan).

A detailed description and instructions about how to download can be found on PhysioBank: https://physionet.org/content/motion-artifact/1.0.0/

We also provide a script (`download_data.sh`) to download the dataset from PhysioBank. To use it, run the following commands:

```bash
cd data/physiobank-motion-artifacts
sh download_data.sh
```

The command requires `wget`.
