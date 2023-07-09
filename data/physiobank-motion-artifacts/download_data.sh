#!/bin/sh
wget -r -nH -l 1 --cut-dirs 3 -np --reject zip https://physionet.org/files/motion-artifact/1.0.0/
