import os
import matplotlib
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
# from word_embeddings_benchmark.web.evaluate import evaluate_on_all
# from word_embeddings_benchmark.web.embeddings import fetch_GloVe, load_embedding

matplotlib.rc('axes', edgecolor='k')


def visualize_embedding_space():
	"""Plot the baseline charts in the paper. Images are written to the img/ subfolder."""
	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:', 'c', 'm', 'y']

	for i, (model, num_layers) in enumerate([('sbert', 13), ('msbert', 13), ('electra', 13), ('simcse', 13), ('roberta', 13), ('bert', 13)]):
		if model == 'ELMo' or model == 'GPT2':
			continue
		x = np.array(range(num_layers))
		data = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
		plt.plot(x, [ data["mean cosine similarity across words"][f'layer_{i}'] for i in x ], icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)
		print(spearmanr(
			[ data["mean cosine similarity across words"][f'layer_{i}'] for i in x ],
			[ data["word norm std"][f'layer_{i}'] for i in x ]
		))

	plt.grid(True, linewidth=0.25)
	plt.legend(loc='upper left')
	plt.xlabel('Layer Index')
	plt.xticks(x)
	plt.ylim(0,1.0)
	plt.title("Average Cosine Similarity between Randomly Sampled Words")
	plt.savefig(f'img/{model}/mean_cosine_similarity_across_words.png', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:', 'c', 'm', 'y']

	for i, (model, num_layers) in enumerate([('sbert', 13), ('msbert', 13), ('electra', 13), ('simcse', 13), ('roberta', 13), ('bert', 13)]):
		if model == 'ELMo' or model == 'GPT2':
			continue
		x = np.array(range(num_layers))
		data = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
		y1 = np.array([ data["mean cosine similarity between sentence and words"][f'layer_{i}'] for i in x ])
		y2 = np.array([ data["mean cosine similarity across words"][f'layer_{i}'] for i in x ])
		plt.plot(x, y1 - y2, icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)

	plt.grid(True, linewidth=0.25)
	plt.legend(loc='upper right')
	plt.xlabel('Layer Index')
	plt.xticks(x)
	plt.ylim(-0.1, 0.5)
	plt.title("Average Intra-Sentence Similarity (anisotropy-adjusted)")
	plt.savefig(f'img/{model}/mean_cosine_similarity_between_sentence_and_words.png', bbox_inches='tight')
	plt.close()


def visualize_self_similarity():
	"""Plot charts relating to self-similarity. Images are written to the img/ subfolder."""
	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:', 'c', 'm', 'y']

	# plot the mean self-similarity but adjust by subtracting the avg similarity between random pairs of words
	for i, (model, num_layers) in enumerate([('sbert', 13), ('msbert', 13), ('electra', 13), ('simcse', 13), ('roberta', 13), ('bert', 13)]):
		if model == 'ELMo' or model == 'GPT2':
			continue
		embedding_stats = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
		self_similarity = pd.read_csv(f'{model.lower()}/self_similarity.csv')

		x = np.array(range(num_layers))
		y1 = np.array([ self_similarity[f'layer_{i}'].mean() for i in x ])
		y2 = np.array([ embedding_stats["mean cosine similarity across words"][f'layer_{i}'] for i in x ])
		plt.plot(x, y1 - y2, icons[i], markersize=6, label=model, linewidth=2.5, alpha=0.65)

	plt.grid(True, linewidth=0.25)
	plt.legend(loc='upper right')
	plt.xlabel('Layer Index')
	plt.xticks(x)
	plt.ylim(0,1)
	plt.title("Average Self-Similarity (anisotropy-adjusted)")
	plt.savefig(f'img/{model}/self_similarity_above_expected.png', bbox_inches='tight')
	plt.close()

	# list the top 10 words that are most self-similar and least self-similar 
	most_self_similar = []
	least_self_similar = []
	models = []

	for i, (model, num_layers) in enumerate([('sbert', 13), ('msbert', 13), ('electra', 13), ('simcse', 13), ('roberta', 13), ('bert', 13)]):
		if model == 'ELMo' or model == 'GPT2':
			continue
		self_similarity = pd.read_csv(f'{model.lower()}/self_similarity.csv')
		self_similarity['avg'] = self_similarity.mean(axis=1)

		models.append(model)
		most_self_similar.append(self_similarity.nlargest(10, 'avg')['word'].tolist())
		least_self_similar.append(self_similarity.nsmallest(10, 'avg')['word'].tolist())
	
	print(' & '.join(models) + '\\\\')
	for tup in zip(*most_self_similar): print(' & '.join(tup) + '\\\\')
	print()
	print(' & '.join(models) + '\\\\')
	for tup in zip(*least_self_similar): print(' & '.join(tup) + '\\\\')


def visualize_variance_explained():
	"""Plot chart for variance explained. Images are written to the img/ subfolder."""
	bar_width = 0.1
	plt.figure(figsize=(12,4))
	icons = [ 'ro:', 'bo:', 'go:', 'c', 'm', 'y']

	# plot the mean variance explained by first PC for occurrences of the same word in different sentences
	# adjust the values by subtracting the variance explained for random sentence vectors
	for i, (model, num_layers) in enumerate([('sbert', 13), ('msbert', 13), ('electra', 13), ('simcse', 13), ('roberta', 13), ('bert', 13)]):
		if model == 'ELMo' or model == 'GPT2':
			continue
		print(model)
		embedding_stats = json.load(open(f'{model.lower()}/embedding_space_stats.json'))
		data = pd.read_csv(f'{model.lower()}/variance_explained.csv')

		x = np.array(range(1, num_layers))

		y1 = np.array([ data[f'layer_{i}'].mean() for i in x ])
		y2 = np.array([ embedding_stats["variance explained for random words"][f'layer_{i}'] for i in x])
		plt.bar(x + i * bar_width, y1 - y2, bar_width, label=model, color=icons[i][0], alpha=0.65)

	plt.grid(True, linewidth=0.25, axis='y')
	plt.legend(loc='upper right')
	plt.xlabel('Layer')
	plt.xticks(x + i * bar_width / 2, x)
	plt.ylim(0,0.5)
	plt.axhline(y=0.05, linewidth=1, color='k', linestyle='--')
	plt.title("Average Maximum Explainable Variance (anisotropy-adjusted)")
	plt.savefig(f'img/{model}/variance_explained.png', bbox_inches='tight')
	plt.show()
	plt.close()


if __name__ == "__main__":

	# EMBEDDINGS_PATH = "/home/keonwoo/anaconda3/envs/DPR/DataMiningTechnique/contextual/contextual_embeddings"
	visualize_variance_explained()
	visualize_self_similarity()
	visualize_embedding_space()