import os, glob
import h5py
from pathlib import Path
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import cPickle
import gopher
import tfomics
import acme
from acme import utils
from tqdm import tqdm

def get_dataset():
    filepath = "/home/rohit/projects/synthetic_runs/data/synthetic/synthetic_code_dataset.h5"
    with h5py.File(filepath, 'r') as dataset:
        x_test = np.array(dataset['X_test']).astype(np.float32)
        y_test = np.array(dataset['Y_test']).astype(np.int32)

    x_test = x_test.transpose([0,2,1])

    N, L, A = x_test.shape

    # get positive samples
    pos_index = np.where(y_test[:,0])[0]
    X = x_test[pos_index]

    return X

def allkeys(obj):
    "Recursively find all keys in an h5py.Group."
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys


file_path = "/home/rohit/projects/synthetic_runs/results/synthetic_attrmaps.h5"
file = h5py.File(file_path, "r")
attr_map_paths = allkeys(file)

# load synthetic sequence
X = get_dataset()

# extract attr_map paths from h5 file
attr_maps = []
count = 0
for path in attr_map_paths:

    if isinstance(file[path], h5py.Dataset):
        attr_maps += [path]


data = {
    "model": [],
    "kld": []
}

for i, attr_map_path in enumerate(attr_maps):
    print(i)
    attr_map = file[attr_map_path][:]

    # normalize attribution map & apply gradient correction
    attr_map = attr_map - np.mean(attr_map, axis=-1, keepdims=True)
    attr_map = attr_map / np.sqrt(np.sum(np.sum(np.square(attr_map), axis=-1, keepdims=True), axis=-2, keepdims=True))

    # calculate kld
    print("Calculating KLD!")
    kld = utils.calculate_kld(
                        sequences=X,
                        attr_maps=attr_map,
                        radius_count_cutoff=0.01
                        )
    data["model"] += [attr_map_path]
    data["kld"] += [kld]

# save the result to a csv
pd.DataFrame(data).to_csv("/home/chandana/projects/acme/results/synthetic/entropy_results.csv")
