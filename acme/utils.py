import os
import numpy as np
import matplotlib.pyplot as plt
import tfomics

def plot_consistency_map(
    sequences,
    attr_map,
    title,
    radius_count_cutoff=0.01,
    numerosity=0,
    colors=["green", "blue", "orange"],
    markersize=1.0,
    alpha=0.9,
    save=False,
    save_path=None
    ):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    radius_count = int(radius_count_cutoff * np.prod(sequences.shape[:2]))
    xxx_motif, yyy_motif, zzz_motif = attr_map[:, :, 0], attr_map[:,:,1], attr_map[:,:,2]   ## (batch, L)
    r = np.linalg.norm(attr_map[:,:, :-1], axis=-1) ## (batch, L)
    x = xxx_motif.flatten() ## (batch*L,)
    print(x.shape)
    y = yyy_motif.flatten()  ## (batch*L,)
    z = zzz_motif.flatten() ## (batch*L,)
    r = r.flatten() ## (batch*L,)
    cutoff = np.sort(r)[-radius_count]
    R_cuttof_index = np.sqrt(x * x + y * y + z * z) > cutoff

    # Cut off
    x = x[R_cuttof_index]
    print(x.shape)
    y = y[R_cuttof_index]
    z = z[R_cuttof_index]
    r = r[R_cuttof_index]


    for color, datum in zip(colors, [x, y, z]):
        plt.plot(x, y, z, '.', color=color, markersize=markersize, alpha=alpha, rasterized=True)
        print(datum.shape)
        # print(color)
        # print(datum)

    ax.set_xlim(-0.33, 0.33)
    ax.set_ylim(-0.33, 0.33)
    ax.set_zlim(-0.33, 0.33)
    ax.view_init(elev=30, azim=45)

    plt.title(title)
    if save:
        fig.savefig(f"{save_path}.pdf", format='pdf', dpi=200, bbox_inches='tight')
    return


# function to plot the saliency maps
def plot_saliency_logos_oneplot(saliency_scores, X_sample, window=20,
                                title=None,
                                filename=None, n_plots=1):
    """
    Function for plotting and saving saliency maps
    :param saliency_scores: pre-computed saliency scores
    :param X_sample: input sequences
    :param window: window around peak saliency to plot
    :param titles: title of each subplot
    :param filename: filepath where the svg will be saved
    :return: None
    """
    N, L, A = X_sample.shape
    fig, axs = plt.subplots(N*n_plots, 1, figsize=[20, 2 * N*n_plots])
    counter = 0
    for i in range(N):

        x_sample = np.expand_dims(X_sample[i], axis=0)
        scores = np.expand_dims(saliency_scores[i], axis=0)
        # find window to plot saliency maps (about max saliency value)
        max_scores = np.max(np.abs(scores), axis=2)[0]
        indices = max_scores.argsort()[-n_plots:][::-1]
        # index = np.argmax(np.max(np.abs(scores), axis=2), axis=1)[0]

        for j, index in enumerate(indices):


            ax = axs[counter]
            if index - window < 0:
                start = 0
                end = window * 2 + 1
            elif index + window > L:
                start = L - window * 2 - 1
                end = L
            else:
                start = index - window
                end = index + window

            saliency_df = tfomics.impress.grad_times_input_to_df(x_sample[:, start:end, :], scores[:, start:end, :])
            tfomics.impress.plot_attribution_map(saliency_df, ax, figsize=(20, 1))

            ax.set_title(f'Sequence index: {i}')
            counter += 1
    fig.suptitle(title)
    if filename:
        if os.path.isfile(filename):
            print('File exists, overwriting results')
        plt.savefig(filename, format='png')

def spherical_coordinate_process(sequences, attr_map, radius_count_cutoff=0.04):
  """
  Calculate the spherical coordinates of points in the acme plot.
  """
  radius_count = int(radius_count_cutoff * np.prod(sequences.shape[:2]))
  xxx_motif, yyy_motif, zzz_motif = attr_map[:, :, 0], attr_map[:,:,1], attr_map[:,:,2]   ## (batch, L)
  r = np.linalg.norm(attr_map[:,:, :-1], axis=-1) ## (batch, L)
  x = xxx_motif.flatten() ## (batch*L,)
  y = yyy_motif.flatten()  ## (batch*L,)
  z = zzz_motif.flatten() ## (batch*L,)
  r = r.flatten() ## (batch*L,)
  cutoff = np.sort(r)[-radius_count]
  R_cuttof_index = np.sqrt(x * x + y * y + z * z) > cutoff

  # Cut off
  x = x[R_cuttof_index]
  y = y[R_cuttof_index]
  z = z[R_cuttof_index]
  r = r[R_cuttof_index]

  # calculate spherical coordinates
  theta = np.arctan(y / x)
  phi = np.arccos(z / r) - np.pi * 0.5
  return theta, phi, r

def initialize_integration(box_length):
    lim = np.pi * 0.5
    box_volume = box_length * box_length
    n_bins = int(2 * lim / box_length)
    n_bins_half = int(n_bins / 2.0)
    return lim, box_length, box_volume, n_bins, n_bins_half

def calc_empirical_box_pdf(theta, phi, box_length = 0.1):
    lim, box_length, box_volume, n_bins, n_bins_half = initialize_integration(box_length)
    n_points = len(theta) # Number of points

    # Now populate the empirical box. Go over every single point.
    empirical_box_count = np.zeros((n_bins, n_bins))
    for i in range (n_points):
        # k, l, m are box numbers of the (theta, phi) point
        idx1 = np.minimum(int(theta[i] / box_length) + n_bins_half, n_bins - 1)
        idx2 = np.minimum(int(phi[i] / box_length) + n_bins_half, n_bins - 1)
        empirical_box_count[idx1, idx2] += 1

    # To get the probability distribution, divide the Empirical_box_count by the total number of points.
    empirical_box_pdf = (empirical_box_count / n_points) / box_volume

    # Check that it integrates to around 1:
    #print('Integral of the empirical_box_pdf (before first renormalization) = ' ,
    #np.sum(empirical_box_pdf*box_volume), '(should be 1.0 if OK) \n')

    # apply correction
    correction = 1 / np.sum(empirical_box_pdf * box_volume)
    empirical_box_pdf = empirical_box_pdf * correction
    return empirical_box_pdf, box_volume

def kl_divergence(empirical_box_pdf, box_volume):
  relative_entropy = 0
  l1, l2 = empirical_box_pdf.shape
  for i in range(l1):
    for j in range(l2):
      if empirical_box_pdf[i, j] > 0:
        phi_angle = np.pi*(j/l1)
        kl_div_contrib = empirical_box_pdf[i,j] * np.log(empirical_box_pdf[i, j]/(np.sin(phi_angle)/(4*np.pi)))
        if np.sin(phi_angle) > 0:
          relative_entropy += kl_div_contrib
  relative_entropy = relative_entropy * box_volume
  return relative_entropy

def calculate_kld(sequences, attr_maps, radius_count_cutoff = 0.04, box_length = 0.1, **kwargs):
  theta, phi, r = spherical_coordinate_process(sequences, attr_maps, radius_count_cutoff)
  res = calc_empirical_box_pdf(theta, phi)
  kl_div = kl_divergence(*res)

  return kl_div
