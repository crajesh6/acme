import os, glob
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import cPickle
import gopher
import tfomics
import acme
from acme import utils

cell_line = "PC-3"
model_name = "blah"
model_path = "/shared/share_zenodo/trained_models/new_models/CNN/1/all/Exp/run-20220321_171001-53za5qem"
data_dir = "/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_13/complete/peak_centered/i_2048_w_1/"
attr_map_path = "/home/amber/saliency_repo/new_models_CNN_1_all_Exp_PC-3.pickle"
radius_count_cutoff = 0.01

# load input sequences
test_set = gopher.utils.make_dataset(
                    data_dir,
                    "test",
                    gopher.utils.load_stats(data_dir),
                    batch_size=128,
                    shuffle=False
                    )
X = np.array([x.numpy() for x, y in test_set.unbatch()])

# load attribution map
with open(attr_map_path, "rb") as input_file:
    attr_map = cPickle.load(input_file)

# normalize attribution map & apply gradient correction
attr_map = attr_map - np.mean(attr_map, axis=-1, keepdims=True)
attr_map = attr_map / np.sqrt(np.sum(np.sum(np.square(attr_map), axis=-1, keepdims=True), axis=-2, keepdims=True))

# calculate kld
kld = utils.calculate_kld(
                    sequences=X,
                    attr_maps=attr_map,
                    radius_count_cutoff=radius_count_cutoff
                    )

# save ACME plot
title = f"{cell_line}; KLD: {kld}; model: {model_name}"
utils.plot_consistency_map(X, attr_map, save=True, title=title, save_path="/home/chandana/projects/acme/results/test_acme")
import h5py
file = h5py.File("/home/rohit/projects/model-selection/notebooks/model.h5", "r")

file
