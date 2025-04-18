import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import pickle

def cal_dloss_inc(potential1: torch.Tensor, potential2: torch.Tensor, point_mass):
	D_real_int = torch.mean(potential1 / point_mass)
	D_fake_int = torch.mean(potential2)
	d_loss = D_fake_int - D_real_int
	return d_loss


def vis(All_sample_r, All_sample_f, All_labels, LOG_folder, source_color='r', target_color='b', name='vis'):

	R = All_sample_r.view(All_sample_r.shape[0], -1)
	F = All_sample_f.view(All_sample_f.shape[0], -1)


	source_feature = F.numpy()
	target_feature = R.numpy()
	All_labels = All_labels.numpy()
	source_len = source_feature.shape[0]
	features = np.concatenate([source_feature, target_feature], axis=0)

	# map features to 2-d using TSNE
	X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

	Outlier_mask = (All_labels == 1)
	Inlier_mask = (All_labels == 0)

	plt.figure(figsize=(10, 10))
	plt.scatter(X_tsne[source_len:][Outlier_mask, 0], X_tsne[source_len:][Outlier_mask, 1], c='gray', s=10, alpha=0.5, marker='s')
	plt.scatter(X_tsne[source_len:][Inlier_mask, 0],  X_tsne[source_len:][Inlier_mask, 1], c=target_color, s=10, alpha=0.5, marker='s')
	plt.scatter(X_tsne[:source_len, 0], X_tsne[:source_len, 1], c=source_color, s=10, alpha=0.3, marker='s')

	tSNE_filename = os.path.join(LOG_folder, '{}_TSNE.png'.format(name))
	plt.savefig(tSNE_filename)

	All_data = {
		'source_feature': source_feature,
		'target_feature': target_feature,
		'All_labels': All_labels,
	}

	with open(os.path.join(LOG_folder, f'{name}.pkl'), 'wb') as f:
		pickle.dump(All_data, f)

