import os, glob
# os.environ['CUDA_VISIBLE_DEVICES'] = '2' # use if not using gpu scheduler
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



@click.command()
@click.option("--cell_line", type=str)
# @click.option("--model_name", type=str)
@click.option("--model_path", type=str)
@click.option("--quantitative_data_dir", type=str)
@click.option("--binary_data_dir", type=str, default="/shared/share_zenodo/datasets/binary_data/peak_center_test.h5")
@click.option("--testset_dir", type=str, default="/shared/share_zenodo/datasets/quantitative_data/testset")
@click.option("--attr_map_path", type=str)
@click.option("--radius_count_cutoff", type=float, default=0.01)
@click.option("--window_size", type=int, default=256)
@click.option("--num_saliency_plots", type=int, default=3)
@click.option("--base_dir", type=str)
@click.option("--model_type", type=str)
@click.option("--evaluate_model", type=bool, default=False)
def main(
    cell_line: str,
    # model_name: str,
    model_path: str,
    quantitative_data_dir: str,
    binary_data_dir: str,
    testset_dir: str,
    attr_map_path: str,
    radius_count_cutoff: float,
    window_size: int,
    num_saliency_plots: int,
    base_dir: str,
    model_type: str,
    evaluate_model: bool
    ):
    model_name = model_path.split('/')[4:-1]
    model_name = '_'.join(model_name)

    attr_map_path = f"{attr_map_path}/{model_name}_{cell_line}.pickle"

    print(f"Starting for {cell_line}!")
    print(base_dir)
    print(f"{model_name}_{cell_line}")

    output_dir = os.path.join(base_dir, f"{model_name}_{cell_line}")
    Path(output_dir).mkdir(exist_ok=True)

    # load input sequences
    if(model_type == "quantitative"):
        test_set = gopher.utils.make_dataset(
                            quantitative_data_dir,
                            "test",
                            gopher.utils.load_stats(quantitative_data_dir),
                            batch_size=128,
                            shuffle=False
                            )
        X = np.array([x.numpy() for x, y in test_set.unbatch()])

    if(model_type == "binary"):
        dataset = h5py.File(binary_data_dir, "r")
        X = dataset["x_test"][:]

    # load attribution map
    with open(attr_map_path, "rb") as input_file:
        attr_map = cPickle.load(input_file)

    # normalize attribution map & apply gradient correction
    attr_map = attr_map - np.mean(attr_map, axis=-1, keepdims=True)
    attr_map = attr_map / np.sqrt(np.sum(np.sum(np.square(attr_map), axis=-1, keepdims=True), axis=-2, keepdims=True))

    # calculate kld
    print("Calculating KLD!")
    kld = utils.calculate_kld(
                        sequences=X,
                        attr_maps=attr_map,
                        radius_count_cutoff=radius_count_cutoff
                        )
    # save kld to csv file
    pd.DataFrame({"kld": [kld]}).to_csv(f"{output_dir}/entropy_result.csv", index=None)

    # evaluate the model -- binary
    if(evaluate_model and model_type == "binary"):
        aupr, auroc = gopher.binary_comparison.binary_metrics(model_path, binary_data_dir)
        pd.DataFrame({
                    "aupr": [aupr],
                    "auroc": [auroc]
                    }).to_csv(f"{output_dir}/evaluation_results.csv", index=None)

    # evaluate the model -- quantitative
    if(evaluate_model and model_type == "quantitative"):
        gopher.evaluate.evaluate_project(
            data_dir=testset_dir,
            run_dir_list=[model_path],
            output_dir=output_dir,
            batch_size=128,
            fast=True
        )
    # save ACME plot
    title = f"{cell_line}; KLD: {kld}; model: {model_name}"
    utils.plot_consistency_map(X, attr_map, save=True, title=title, save_path=f"{output_dir}/test_acme")

    # get first N sample sequences
    sample_indices = np.arange(num_saliency_plots)
    X_sub = X[sample_indices]

    utils.plot_saliency_logos_oneplot(
        attr_map,
        X_sub,
        window=window_size,
        title=f"{model_name} {cell_line}",
        filename=f"{output_dir}/{model_name}_{cell_line}_saliency_map.png"
    )
    # save the input arguments into a DataFrame
    # TODO

if __name__ == "__main__":
    main()
