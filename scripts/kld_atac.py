import os, glob
import h5py
from pathlib import Path
import sys
import time
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt



@click.command()
@click.option("--cell_line", type=str)
@click.option("--model_path", type=str)
@click.option("--data_dir", type=str)
@click.option("--binary_testset_dir", type=str, default="/shared/share_zenodo/datasets/binary_data/peak_center_test.h5")
@click.option("--quantitative_testset_dir", type=str, default="/shared/share_zenodo/datasets/quantitative_data/testset")
@click.option("--attr_map_path", type=str)
@click.option("--radius_count_cutoff", type=float, default=0.01)
@click.option("--window_size", type=int, default=256)
@click.option("--num_saliency_plots", type=int, default=3)
@click.option("--base_dir", type=str)
@click.option("--model_type", type=str)
@click.option("--evaluate_model", type=bool, default=False)
@click.option("--plot_acme", type=bool, default=False)
@click.option("--gpu", type=str, default="2")
def main(
    cell_line: str,
    model_path: str,
    data_dir: str,
    binary_testset_dir: str,
    quantitative_testset_dir: str,
    attr_map_path: str,
    radius_count_cutoff: float,
    window_size: int,
    num_saliency_plots: int,
    base_dir: str,
    model_type: str,
    evaluate_model: bool,
    plot_acme: bool,
    gpu: str
    ):

    start = time.time()

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu # use if not using gpu scheduler
    # create base directory if it doesn't exist
    Path(f"{base_dir}").mkdir(parents=True, exist_ok=True)

    model_name = model_path.split('/')[4:-1]
    model_name = '_'.join(model_name)

    attr_map_path = f"{attr_map_path}/{model_name}_{cell_line}.pickle"

    print(f"Starting for {cell_line}!")

    # create output directory if it doesn't exist
    output_dir = os.path.join(base_dir, f"{model_name}_{cell_line}")
    Path(output_dir).mkdir(exist_ok=True)

    # load cell line specific input sequences
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
    print("Calculating KLD!")
    kld = utils.calculate_kld(
                        sequences=X.copy(),
                        attr_maps=attr_map.copy(),
                        radius_count_cutoff=radius_count_cutoff
                        )
    # save kld to csv file
    pd.DataFrame({"kld": [kld]}).to_csv(f"{output_dir}/entropy_result.csv", index=None)

    # evaluate the model -- binary
    if(evaluate_model and model_type == "binary"):
        aupr, auroc = gopher.binary_comparison.binary_metrics(model_path, binary_testset_dir)
        pd.DataFrame({
                    "aupr": [aupr],
                    "auroc": [auroc]
                    }).to_csv(f"{output_dir}/evaluation_results.csv", index=None)

    # evaluate the model -- quantitative
    if(evaluate_model and model_type == "quantitative"):
        gopher.evaluate.evaluate_project(
            data_dir=quantitative_testset_dir,
            run_dir_list=[model_path],
            output_dir=output_dir,
            batch_size=128,
            fast=True
        )
    # save ACME plot
    if(plot_acme):
        title = f"{cell_line}; KLD: {kld}; RCC: {radius_count_cutoff}; model: {model_name}"
        utils.plot_consistency_map(
                    X.copy(),
                    attr_map.copy(),
                    save=True,
                    title=title,
                    radius_count_cutoff=radius_count_cutoff,
                    save_path=f"{output_dir}/{model_name}_{cell_line}_acme"
                    )

    # get first N sample sequences
    # sample_indices = np.arange(num_saliency_plots)
    # TODO -- look up how these SHOULD be plotted
    # X_sub = X[0]
    # attr_map_sub = attr_map[0]
    # print(attr_map_sub[0])

    # utils.plot_saliency_logos_oneplot(
    #     attr_map_sub,
    #     X_sub,
    #     window=window_size,
    #     title=f"{model_name} {cell_line}",
    #     filename=f"/home/chandana/projects/acme/results/saliency_map_shamber.png"
    # )

    utils.plot_sequence_logomaker(attr_map.copy(), s=0, image_length=25, image_height=2, save_path=f"{output_dir}/saliency_map.png")

    # save the input arguments into a log file
    logfile = open(f"{output_dir}/logfile.txt", "w")
    logfile.write(" ".join(sys.argv))
    logfile.write(f" {attr_map_path}")
    logfile.close()

    end = time.time()
    print(f"Done with KLD evaluation for {model_name} and {cell_line}!")
    print(f"Time taken: {end - start} seconds")

if __name__ == "__main__":
    main()
