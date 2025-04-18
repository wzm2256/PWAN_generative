import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def vis(file_path, source_color='r', target_color='b', name='vis'):

	with open(file_path, 'rb') as f:
		All_data = pickle.load(f)

	source_feature = All_data['source_feature']
	target_feature = All_data['target_feature']
	All_labels = All_data['All_labels']
	
	
	source_len = source_feature.shape[0]
	features = np.concatenate([source_feature, target_feature], axis=0)

	#######
	X_tsne = All_data['X_tsne']
	#######
	# X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
	#######

	Outlier_mask = (All_labels == 1)
	Inlier_mask = (All_labels == 0)

	plt.figure(figsize=(5, 5))
	plt.scatter(X_tsne[source_len:][Outlier_mask, 0], X_tsne[source_len:][Outlier_mask, 1], c='black', s=5, alpha=1.0, marker='o', label='MNIST')
	plt.scatter(X_tsne[source_len:][Outlier_mask, 0], X_tsne[source_len:][Outlier_mask, 1], c='black', s=15, alpha=1.0, marker='o')
	plt.scatter(X_tsne[source_len:][Inlier_mask, 0],  X_tsne[source_len:][Inlier_mask, 1], c=target_color, s=5, alpha=1.0, marker='o', label='Cifar10')
	plt.scatter(X_tsne[:source_len, 0], X_tsne[:source_len, 1], c=source_color, s=5, alpha=0.5, marker='o', label='Generated')
	plt.scatter(X_tsne[:source_len, 0], X_tsne[:source_len, 1], c=source_color, s=5, alpha=0.5, marker='o')

	tSNE_filename = os.path.join(os.path.dirname(file_path), f'{os.path.basename(file_path).split(".")[0]}_TSNE.png')
	# plt.axis('off')
	plt.xticks([])
	plt.yticks([])

	fontsize = 25
	fontsizex = 20
	fontlabel = 25
	linewidth = 2
	size = 8
	elinewidth = 2.


	plt.legend(fontsize=fontsize, loc=1, markerscale=4, handlelength=0.5)

	plt.tight_layout()

	###########
	# plt.show()
	###########
	plt.savefig(tSNE_filename)
	###########

	# All_data = {
	# 	'source_feature': source_feature,
	# 	'target_feature': target_feature,
	# 	'All_labels': All_labels,
	# 	'X_tsne': X_tsne
	# }
	#
	# with open(os.path.join(os.path.dirname(file_path), 'new_'+ os.path.basename(file_path)), 'wb') as f:
	# 	pickle.dump(All_data, f)



if __name__ == '__main__':
	import argparse

	args_ps = argparse.ArgumentParser()
	args_ps.add_argument('file_path', type=str)
	args = args_ps.parse_args()

	vis(args.file_path, source_color='r', target_color='b', name='vis')
