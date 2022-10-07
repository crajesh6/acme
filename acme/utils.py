import glob, h5py, os, re, sys, time
from itertools import product
from pathlib import Path
from filelock import FileLock

import matplotlib.pyplot as plt
import logomaker
import tfomics
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from multiprocess import Pool
from tqdm.notebook import tqdm
from progressbar import ProgressBar
from scipy.special import softmax, kl_div, rel_entr
from six.moves import cPickle
from tqdm import tqdm

from acme.kmer import kmer_featurization
from acme import interval


BASE_DIR = Path.cwd().parent
evaluation_path = BASE_DIR.joinpath("data/atac/atac_model_pearson.csv")

DATA_DIR = BASE_DIR.joinpath("data/atac/cell_line_testsets")
saliency_dir = BASE_DIR.joinpath("data/atac/saliency_repo")
# cell line paths
cell_line_dict = {
    "A549": f"{DATA_DIR}/cell_line_8.h5",
    "HCT116": f"{DATA_DIR}/cell_line_9.h5",
    "GM12878": f"{DATA_DIR}/cell_line_7.h5",
    "K562": f"{DATA_DIR}/cell_line_5.h5",
    "PC-3": f"{DATA_DIR}/cell_line_13.h5",
    "HepG2": f"{DATA_DIR}/cell_line_2.h5"
}

# model dict
model_dict = {
    "new_models_Residual_32_task_Exp": "residual_32_task_exp",
    "new_models_CNN_1_all_Exp": "cnn_base_all_exp",
    "new_models_Residual_32_all_Exp": "residual_32_all_exp",
    "new_models_CNN_32_all_Exp": "cnn_32_all_exp",
    "new_models_Residual_32_task_ReLU": "residual_32_task_relu",
    "binary_basenji_binary_exp": "binary_basenji_exp",
    "bpnet_augmentation_48": "bpnet",
    "binary_residual_binary": "binary_residual_relu",
    "new_models_Residual_1_task_Exp": "residual_base_task_exp",
    "binary_basenji_binary": "binary_basenji_relu",
    "new_models_CNN_32_task_Exp": "cnn_32_task_exp",
    "new_models_Residual_1_all_Exp": "residual_base_all_exp",
    "binary_residual_binary_exp": "binary_residual_exp",
    "new_models_CNN_1_task_ReLU": "cnn_base_task_relu",
    "binary_conv_binary_exp": "binary_cnn_exp",
    "binary_basset_exp": "binary_basset_exp",
    "new_models_Residual_1_all_ReLU": "residual_base_all_relu",
    "binary_basset": "binary_basset_relu",
    "new_models_CNN_32_task_ReLU": "cnn_32_task_relu",
    "new_models_CNN_32_all_ReLU": "cnn_32_all_relu",
    "new_models_CNN_1_task_Exp": "cnn_base_task_exp",
    "new_models_Residual_1_task_ReLU": "residual_base_task_relu",
    "basenji_v2_binloss_basenji_v2": "basenji_v2_binloss_relu",
    "new_models_Residual_32_all_ReLU": "residual_32_all_relu",
    "new_models_CNN_1_all_ReLU": "cnn_base_all_relu",
    "binary_conv_binary": "binary_cnn_relu"
}

##############################################################################
# DATASET LOADING
##############################################################################

def load_data(
        attr_map_path: str,
        cell_line_dir: str,
        gradient_correct: bool = True,
        normalize: bool = True
    ) -> (np.ndarray, np.ndarray):
    """Load dataset and attribution map"""

    # load attribution map
    with FileLock(os.path.expanduser(f"{attr_map_path}.lock")):
        with open(attr_map_path, "rb") as input_file:
            attr_map = cPickle.load(input_file)[:]

    # normalize attribution map & apply gradient correction
    if gradient_correct:
        attr_map = attr_map - attr_map.mean(-1, keepdims=True)
    if normalize:
        attr_map = attr_map / np.sqrt(np.sum(np.sum(np.square(attr_map), axis=-1, keepdims=True), axis=-2, keepdims=True))

    # load test set
    with FileLock(os.path.expanduser(f"{cell_line_dir}.lock")):
        with h5py.File(cell_line_dir, 'r') as hf:
            X = hf["X"][:]
            y = hf["y"][:]

    return attr_map, X, y



def get_model_info(saliency_dir: str):
    """Create dataframe containing info of models and cell lines"""

    data_paths = glob.glob(f"{saliency_dir}/*/*/*/*.pickle")

    data = {
        "model": [],
        "cell_line": [],
        "cell_line_dir": [],
        "attr_map_path": [],
        "task_type": [],
        "activation": [],
    }

    for data_path in data_paths:

        attr_map_path = data_path
        model = data_path.split("/")[-1].split(".pickle")[0]
        activation = data_path.split("/")[-2]
        task_type = data_path.split("/")[-3]
        cell_line = data_path.split("/")[-4]
        cell_line_dir = cell_line_dict[cell_line]

        data["model"] += [model]
        data["cell_line"] += [cell_line]
        data["cell_line_dir"] += [cell_line_dir]
        data["attr_map_path"] += [attr_map_path]
        data["task_type"] += [task_type]
        data["activation"] += [activation]

    df = pd.DataFrame(data)

    return df



def matrix_to_df(x, w, alphabet='ACGT'):
    """generate pandas dataframe for saliency plot
     based on grad x inputs """

    L, A = w.shape
    counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(L)))
    for a in range(A):
        for l in range(L):
            counts_df.iloc[l,a] = w[l,a]
    return counts_df

def plot_attribution_map(saliency_df, ax=None, title=None, figsize=(20,1), fontsize=16):
    """plot an attribution map using logomaker"""

    logomaker.Logo(saliency_df, figsize=figsize, ax=ax)
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_xticks([])

    if(title):
        ax.set_title(title)


def consecutive(data, stepsize=1):
    """split a list of indices into contiguous chunks"""
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def orthonormal_coordinates(attr_map):
    """reduce 4d array to 3d"""

    attr_map_on = np.zeros((attr_map.shape[0], attr_map.shape[1], 3))

    x = attr_map[:, :, 0]
    y = attr_map[:, :, 1]
    z = attr_map[:, :, 2]
    w = attr_map[:, :, 3]

    # Now convert to new coordinates
    e1 = 1 / np.sqrt(2) * (-x + y)
    e2 = np.sqrt(2 / 3) * (-1/2*x -1/2*y)
    e3 = np.sqrt(3 / 4) * (-1/3*x -1/3*y -1/3*z + w)
    attr_map_on[:, :, 0] = e1
    attr_map_on[:, :, 1] = e2
    attr_map_on[:, :, 2] = e3

    return attr_map_on



def plot_kmer_frequency(
        entropy: np.float64,
        obj,
        global_counts_normalized: np.ndarray,
        kmer_prior,
        kmer_dict: dict,
        title: str,
        save_path: str
    ):
    """Plot kmer frequency"""

    xs = np.arange(obj.n)
    ys = global_counts_normalized

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(1,1,1)
    plt.plot(xs,ys, "-o")
    plt.plot(kmer_prior, linewidth=0.5)
    plt.ylabel('Frequency', fontsize=14)
    plt.xlabel('kmer', fontsize=14)
    ax.set_xticks([])

    if(title):
        plt.title(title, fontsize=14)

    # plot annotations
    t = ys[np.argsort(ys)[::-1][5]]
    for i,(x,y) in enumerate(zip(xs,ys)):

        if (y > t):
            label = kmer_dict[i]

            plt.annotate(label,
                         (x,y),
                         textcoords="offset points",
                         xytext=(0,5),
                         ha='center')

    # plt.show()
    if(save_path):
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    return


def compute_kmer_spectra_testset(
    X,
    kmer_length=3,
    dna_dict = {
        0: "A",
        1: "C",
        2: "G",
        3: "T"
      }
    ):
    # convert one hot to A,C,G,T
    seq_list = []

    for index in tqdm(range(len(X))):

        seq = X[index]

        seq_list += ["".join([dna_dict[np.where(i)[0][0]] for i in seq])]


    obj = kmer_featurization(kmer_length)  # initialize a kmer_featurization object
    kmer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(seq_list, write_number_of_occurrences=True)

    kmer_permutations = ["".join(p) for p in product(["A", "C", "G", "T"], repeat=kmer_length)]

    kmer_dict = {}
    for kmer in kmer_permutations:
        n = obj.kmer_numbering_for_one_kmer(kmer)
        kmer_dict[n] = kmer

    global_counts = np.sum(np.array(kmer_features), axis=0)
    # what to compute entropy against
    global_counts_normalized = global_counts / sum(global_counts) # this is the distribution of kmers in the testset

    return global_counts_normalized


def run_pool(my_func, args, n_workers):

    # run using multiprocessing
    print(f"Running function using {n_workers} workers")
    t1 = time.time()
    with Pool(n_workers) as p:
        r = list(tqdm(p.imap(my_func, args), total=len(args)))
    t2 = time.time()

    print(f"Time taken: {np.round(t2 - t1, 2)} seconds!")
    return


# vectorized method
def add_interval_buffer(left, right, sequence_length, buffer_size=2):
    if (right + buffer_size >= sequence_length):
        right = sequence_length
    else:
        right = right + buffer_size
    return interval.Interval(left, right)


def merged_intervals(indices, sequence_length, buffer_size=2):

    intervals = [add_interval_buffer(i[0], i[-1], sequence_length) for i in consecutive(indices)]
    return interval.mergeIntervals(intervals)[::-1]


def aggregate_kmers(X, passing_sequences, kmer_length):

    seq_blocks = []
    for i, _ in enumerate(passing_sequences):

        subseq = passing_sequences[i]
        seq_blocks += [X[i][j] for j in subseq if len(j) >= kmer_length]

    return seq_blocks


def matrix_to_seq(
        seq_block,
        dna_dict = {
            0: "A",
            1: "C",
            2: "G",
            3: "T"
          }
     ):

    return "".join([dna_dict[np.where(i)[0][0]] for i in seq_block])


def collect_passing_subsequences(
    attr_map,
    X,
    threshold=0.9,
    kmer_length=3,
    buffer_size=2,
    ):
    N, L, A = X.shape
    r = np.linalg.norm(attr_map, axis=-1)
    cutoff_array = np.quantile(r, threshold, axis=-1)

    indices = np.argwhere(r > np.expand_dims(cutoff_array, axis=-1))

    split_indices = np.split(indices[:,1], np.unique(indices[:, 0], return_index=True)[1][1:])
    passing_sequences = [merged_intervals(indices, sequence_length=L-1) for i, indices in enumerate(split_indices)]

    seq_blocks = aggregate_kmers(X, passing_sequences, kmer_length=kmer_length)
    seq_list = [matrix_to_seq(i) for i in seq_blocks]

    return seq_list, cutoff_array


def calculate_kmer_entropy(X, seq_list, kmer_prior, kmer_length=3):

    obj = kmer_featurization(kmer_length)  # initialize a kmer_featurization object
    kmer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(seq_list, write_number_of_occurrences=True)

    kmer_permutations = ["".join(p) for p in product(["A", "C", "G", "T"], repeat=kmer_length)]

    kmer_dict = {}
    for kmer in kmer_permutations:
        n = obj.kmer_numbering_for_one_kmer(kmer)
        kmer_dict[n] = kmer

    global_counts = np.sum(np.array(kmer_features), axis=0)

    global_counts_normalized = global_counts / sum(global_counts) # softmax(global_counts)

    entropy = np.round(np.sum(kl_div(global_counts_normalized, kmer_prior)), 3)

    return entropy, obj, global_counts_normalized, kmer_prior, kmer_dict


def my_func_kmer(model_and_attr_map):

    entropy_dict = {
            "model": [],
            "cell_line": [],
            "task_type": [],
            "activation": [],
            "threshold": [],
            "kmer_length": [],
            "entropy": [],
            "pearson_corr": [],
            "n_intervals": [],
            "n_nucleotides": []
        }

    eval_df = pd.read_csv(evaluation_path)

    cell_line_dir, cell_line, model, attr_map_path, task_type, activation, out_path, threshold, kmer_length = model_and_attr_map

    try:
        attr_map, X = load_data(attr_map_path, cell_line_dir) # shape: (N, L, 4)
    except:
        print("FAILED")
        return
    seq_list, cutoff_array = collect_passing_subsequences(attr_map, X, threshold=threshold, kmer_length=kmer_length)

    n_intervals = len(seq_list)
    n_nucleotides = sum([len(i) for i in seq_list])

    kmer_path = f"data/atac/kmer_prior/{cell_line}"
    with FileLock(os.path.expanduser(f"{kmer_path}/kmer_prior.h5.lock")):
        with h5py.File(f"{kmer_path}/kmer_prior.h5", "r") as f:
            kmer_prior = f[f"{kmer_length}"][:]

    entropy, obj, global_counts_normalized, kmer_prior, kmer_dict = calculate_kmer_entropy(
        X,
        seq_list,
        kmer_prior=kmer_prior,
        kmer_length=kmer_length
    )

    entropy_dict["model"] += [model]
    entropy_dict["entropy"] += [entropy]

    pr_corr = eval_df[eval_df["model_name"] == model][cell_line].to_numpy()[0]
    entropy_dict["pearson_corr"] += [pr_corr]
    entropy_dict["cell_line"] += [cell_line]
    entropy_dict["task_type"] += [task_type]
    entropy_dict["activation"] += [activation]
    entropy_dict["threshold"] += [threshold]
    entropy_dict["kmer_length"] += [kmer_length]
    entropy_dict["n_intervals"] += [n_intervals]
    entropy_dict["n_nucleotides"] += [n_nucleotides]

    title = f'{cell_line}; {model}; kmer_length: {kmer_length}; Percentile: {threshold}; Entropy: {entropy}; n_intervals={n_intervals}; n_nucleotides: {n_nucleotides}'
    save_path = f"{out_path}/{model}_{cell_line}_{kmer_length}_kmer_frequency_plot.pdf"
    plot_kmer_frequency(entropy, obj, global_counts_normalized, kmer_prior, kmer_dict, title, save_path)

    entropy_df = pd.DataFrame(entropy_dict)
    # save the dataframe
    pd.DataFrame(entropy_df).to_csv(f"{out_path}/{model}_{cell_line}_{threshold}_{kmer_length}_entropy_results.csv", index=None)

    return


##############################################################################
# PLOTTING FUNCTIONS
##############################################################################

def plot_attribution_map(saliency_df, ax=None, title=None, figsize=(20,1), fontsize=16):
    """plot an attribution map using logomaker"""

    logomaker.Logo(saliency_df, figsize=figsize, ax=ax)
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_xticks([])

    if(title):
        ax.set_title(title)

def _stack_saliency_plots(fig, fig_size, index, attr_map_list, model_name_list, plot_range_list, cell_line_dir, input_mask=False): # cutoff_list=cutoff_list,


    num_models = len(attr_map_list)
    # fig = plt.figure(figsize=(10, 15 * num_models))
    for i, attr_map_path in enumerate(attr_map_list):
        print(i)

        attr_map, X, y = load_data(attr_map_path, cell_line_dir)
        sal_scores = np.expand_dims(attr_map[index], axis=0)
        x = np.expand_dims(X[index], axis=0)

        if(input_mask):

            saliency_df = tfomics.impress.grad_times_input_to_df(x, sal_scores[0, plot_range_list[i],:])

        else:
            saliency_df = tfomics.impress.matrix_to_df(x, sal_scores[0, plot_range_list[i],:])


        ax = plt.subplot(num_models, 1, i+1)
        plot_attribution_map(saliency_df, ax, figsize=fig_size, fontsize=14)
        # plt.plot([cutoff]*len(plot_range_list[i]))
        # plt.plot([cutoff]*len(plot_range))
        name = "\n".join(model_name_list[i].split("_"))
        ax.set_ylabel(f'{name}', fontsize=8)
        ax.set_xlabel(f"Index: {index}", fontsize=5)
#         plt.yticks([-2.,0,2.])
        # ax.settitle("blah")
        # plt.title(f"Index: {index}")

    # plt.tight_layout()
    # plt.show()

    return


def plot_saliency_logos_pdf(
    file_name,
    index_list,
    attr_map_list,
    model_name_list,
    # cutoff_list,
    plot_range_list,
    cell_line_dir,
    fig_size,
    input_mask=False
    ):

#     print(fig_size)

#     pdf = PdfPages(f'{out_path}/stacked_saliency_plots_limit.pdf')
    pdf = PdfPages(f'{file_name}.pdf')
    num_models = len(attr_map_list)
    # fig = plt.figure(figsize=(10, 15 * num_models))
    for index in index_list:


        fig = plt.figure(figsize=(fig_size[0], fig_size[1] * num_models))

        _stack_saliency_plots(fig=fig, fig_size=fig_size, index=index, attr_map_list=attr_map_list, model_name_list=model_name_list, plot_range_list=plot_range_list, cell_line_dir=cell_line_dir, input_mask=input_mask) #cutoff_list=cutoff_list,

        # fig.set_title(f'{index}', fontsize=10)
        # fig.suptitle(f'{index}', fontsize=10, y=.5)

        pdf.savefig(fig, dpi=200, bbox_inches='tight')

        plt.close()
    pdf.close()

    return

##############################################################################
# SYNTHETIC DATASET FUNCTIONS
##############################################################################

def get_dataset():
    filepath = BASE_DIR.joinpath("data/synthetic/synthetic_code_dataset.h5")

    with h5py.File(filepath, 'r') as dataset:
        x_train = np.array(dataset['X_train']).astype(np.float32)
        y_train = np.array(dataset['Y_train']).astype(np.float32)
        x_valid = np.array(dataset['X_valid']).astype(np.float32)
        y_valid = np.array(dataset['Y_valid']).astype(np.int32)
        x_test = np.array(dataset['X_test']).astype(np.float32)
        y_test = np.array(dataset['Y_test']).astype(np.int32)
        model_test = np.array(dataset['model_test']).astype(np.float32)

    model_test = model_test.transpose([0,2,1])
    x_train = x_train.transpose([0,2,1])
    x_valid = x_valid.transpose([0,2,1])
    x_test = x_test.transpose([0,2,1])

    N, L, A = x_train.shape

    # get positive samples
    pos_index = np.where(y_test[:,0])[0]
    X = x_test[pos_index]
    X_model = model_test[pos_index]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), X, X_model

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

def calculate_interp_perf(grad, X, X_model, threshold=0.1, top_k=10):
    res = {}
    grad_times_input = tfomics.explain.grad_times_input(X, grad)
    roc, pr = tfomics.evaluate.interpretability_performance(grad_times_input, X_model, threshold)
    res['roc'] = np.mean(roc)
    res['pr'] = np.mean(pr)
    signal, noise_max, noise_mean, noise_topk = tfomics.evaluate.signal_noise_stats(grad_times_input, X_model, top_k, threshold)
    snr = tfomics.evaluate.calculate_snr(signal, noise_topk)
    res['snr'] = np.nanmean(snr)
    return res
