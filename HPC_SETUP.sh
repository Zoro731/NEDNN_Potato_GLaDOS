#!/bin/bash

# -- link data to the data folder of this repository

workdir="$PWD"

echo ">>> creating links to datasets in $workdir/data"
ln -s /share/klab/danthes/MEG_Encoding-sose2026/shared_data/brainencoding26 ./data/
ln -s /share/klab/danthes/MEG_Encoding-sose2026/shared_data/avs_scenes ./data/
