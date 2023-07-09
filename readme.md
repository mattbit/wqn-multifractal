# The WQN algorithm for EEG artifact removal in the absence of scale invariance 

Supporting code for the article "The WQN algorithm for EEG artifact removal in the absence of scale invariance" by Matteo Dora, Stéphane Jaffard, David Holcman.

## Setup

The code was written for Python 3.11 (but can probably run on previous versions without modification). The dependencies can be installed with `pip`:

```sh
pip install -r requirements.txt
```

## Figures and results

All figures and results in the article can be reproduced by running the scripts in the root folder, which are numbered based on their appearance in the manuscript.

The generated output files will be placed in the `./output` folder.

For example, to reproduce Figure 1, run:

```sh
python 10_multifractal_analysis.py
```

This will generate the file `fig_scale_invariance.pdf` in the `./output` folder, which corresponds to Figure 1 in the article.

**Note:** Not all data required to run the scripts are included in this repository. Each dataset in the `data` folder has a `readme.md` file with instructions on how to download.
Once data is downloaded, run the script `00_preprocess_data.py` to convert the data to the format used by the other scripts.

## Help

If you need help running the code don't hesitate to contact the author at matteo.dora@ieee.org.
