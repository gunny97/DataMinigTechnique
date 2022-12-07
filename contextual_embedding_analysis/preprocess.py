import os
import csv
import json
import spacy
from spacy.lang.ko import Korean
from kobert_tokenizer import KoBertTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoTokenizer

from typing import Dict, Tuple, Sequence, List, Callable

# nlp = spacy.load("ko_core_news_sm")
nlp = Korean()

# spacy_tokenizer = Korean().Defaults.create_tokenizer(nlp)
spacy_tokenizer = nlp.tokenizer

import numpy
import pandas as pd
import torch
import h5py
from pytorch_pretrained import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
# from allennlp.commands.elmo import ElmoEmbedder
# from allennlp.data.tokenizers.token import Token
# from allennlp.common.tqdm import Tqdm
from tqdm import tqdm

class Vectorizer:
	"""
	Abstract class for creating a tensor representation of size (#layers, #tokens, dimensionality)
	for a given sentence.
	"""
	def vectorize(self, sentence: str) -> numpy.ndarray:
		"""
		Abstract method for tokenizing a given sentence and return embeddings of those tokens.
		"""
		raise NotImplemented

	def make_hdf5_file(self, sentences: List[str], out_fn: str) -> None:
		"""
		Given a list of sentences, tokenize each one and vectorize the tokens. Write the embeddings
		to out_fn in the HDF5 file format. The index in the data corresponds to the sentence index.
		"""
		sentence_index = 0

		with h5py.File(out_fn, 'w') as fout:
			for sentence in tqdm(sentences):
				embeddings = self.vectorize(sentence)
				fout.create_dataset(str(sentence_index), embeddings.shape, dtype='float32', data=embeddings)
				sentence_index += 1


# class ELMo(Vectorizer):
# 	def __init__(self):
# 		self.elmo = ElmoEmbedder()

# 	def vectorize(self, sentence: str) -> numpy.ndarray:
# 		"""
# 		Return a tensor representation of the sentence of size (3 layers, num tokens, 1024 dim).
# 		"""
# 		# tokenizer's tokens must be converted to string tokens first
# 		tokens = list(map(str, spacy_tokenizer(sentence)))	
# 		embeddings = self.elmo.embed_sentence(tokens)
# 		return embeddings 	


class BertBaseCased(Vectorizer):
	def __init__(self, ckpt):
		# self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
		# self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
		self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
		self.ckpt = ckpt
		# self.model = BertModel.from_pretrained('bert-base-cased')
		# config = AutoConfig.from_pretrained("monologg/kobert", output_hidden_states=True)
		# self.model = AutoModel.from_pretrained('monologg/kobert', config=config)

		config = AutoConfig.from_pretrained(ckpt, output_hidden_states=True)
		self.model = AutoModel.from_pretrained(ckpt, config=config)
		self.model.eval()

	BertModel
	def vectorize(self, sentence: str) -> numpy.ndarray:
		"""
		Return a tensor representation of the sentence of size (13 layers, num tokens, 768 dim).
		Even though there are only 12 layers in GPT2, we include the input embeddings as the first
		layer (for a fairer comparison to ELMo).
		"""
		# add CLS and SEP to mark the start and end
		tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]']
		# tokenize sentence with custom BERT tokenizer
		token_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
		# segment ids are all the same (since it's all one sentence)
		segment_ids = numpy.zeros_like(token_ids)

		tokens_tensor = torch.tensor([token_ids])

		

		segments_tensor = torch.tensor([segment_ids])

		with torch.no_grad():
			# embeddings, _, input_embeddings = self.model(tokens_tensor, segments_tensor)
			
			# electra
			# input_embeddings, embeddings = self.model(tokens_tensor, segments_tensor)[0], self.model(tokens_tensor, segments_tensor)[1][1:]
			
			# other encoders
			input_embeddings, embeddings = self.model(tokens_tensor, segments_tensor)[0], self.model(tokens_tensor, segments_tensor)[2][1:]

		# exclude embeddings for CLS and SEP; then, convert to numpy

		embeddings = torch.stack([input_embeddings] + list(embeddings), dim=0).squeeze()[:,1:-1,:]
		# embeddings = embeddings[:,1:-1,:]
		embeddings = embeddings.detach().numpy()							

		return embeddings

def index_tokens(tokens: List[str], sent_index: int, indexer: Dict[str, List[Tuple[int, int]]]) -> None:
	"""
	Given string tokens that all appear in the same sentence, append tuple (sentence index, index of
	word in sentence) to the list of values each token is mapped to in indexer. Exclude tokens that 
	are punctuation.

	Args:
		tokens: string tokens that all appear in the same sentence
		sent_index: index of sentence in the data
		indexer: map of string tokens to a list of unique tuples, one for each sentence the token 
			appears in; each tuple is of the form (sentence index, index of token in that sentence)
	"""
	for token_index, token in enumerate(tokens):
		if not nlp.vocab[token].is_punct:
			if str(token) not in indexer:
				indexer[str(token)] = []

			indexer[str(token)].append((sent_index, token_index))


def index_sentence(data_fn: str, index_fn: str, tokenize: Callable[[str], List[str]], min_count=5) -> List[str]:
	"""
	Given a data file data_fn with the format of sts.csv, index the words by sentence in the order
	they appear in data_fn. 

	Args:
		index_fn: at index_fn, create a JSON file mapping each word to a list of tuples, each 
			containing the sentence it appears in and its index in that sentence
		tokenize: a callable function that maps each sentence to a list of string tokens; identity
			and number of tokens generated can vary across functions
		min_count: tokens appearing fewer than min_count times are left out of index_fn

	Return:
		List of sentences in the order they were indexed.
	"""
	word2sent_indexer = {}
	sentences = []
	sentence_index = 0

	with open(data_fn) as csvfile:
		# csvreader = csv.DictReader(csvfile, quotechar='"', delimiter='\t')
		csvreader = csv.DictReader(csvfile, quotechar='"')
		
		for line in csvreader:
			# only consider scored sentence pairs
			if line['Score'] == '':	
				continue

			# handle case where \t is between incomplete quotes (causes sents to be treated as one)
			if line['Sent2'] is None:
				line['Sent1'], line['Sent2'] = line['Sent1'].split('\t')[:2]

			index_tokens(tokenize(line['Sent1']), sentence_index, word2sent_indexer)
			index_tokens(tokenize(line['Sent2']), sentence_index + 1, word2sent_indexer)
			sentences.append(line['Sent1'])
			sentences.append(line['Sent2'])
			sentence_index += 2

	# remove words that appear less than min_count times
	infrequent_words = list(filter(lambda w: len(word2sent_indexer[w]) < min_count, word2sent_indexer.keys()))
	
	for w in infrequent_words:
		del word2sent_indexer[w]

	json.dump(word2sent_indexer, open(index_fn, 'w'), indent=1)
	
	return sentences


if __name__ == "__main__":
	# where to save the contextualized embeddings
	EMBEDDINGS_PATH = "/home/keonwoo/anaconda3/envs/DPR/DataMiningTechnique/contextual/contextual_embeddings"

	# sts.csv has been preprocessed to remove all quotes of type ", since they are often not completed
	# elmo = ELMo()
	# sentences = index_sentence('sts.csv', 'elmo/word2sent.json', lambda s: list(map(str, spacy_tokenizer(s))))
	# elmo.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'elmo.hdf5'))

	path = "/home/keonwoo/anaconda3/envs/KoDiffCSE/data/ko_sts_test.txt"

	data = pd.read_csv(path)
	data = data[['score','sentence1','sentence2']]
	data.columns = ['Score','Sent1','Sent2']
	data.to_csv('ko_sts.csv')

	# print('start BERT')
	# bert = BertBaseCased()
	# sentences = index_sentence('ko_sts.csv', 'bert/word2sent.json', bert.tokenizer.tokenize)
	# print('clear')
	# bert.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, 'bert.hdf5'))
	# print('finish BERT')

	ko_sbert_multitask = ('jhgan/ko-sbert-multitask', 'sbert')
	msbert = ('sentence-transformers/paraphrase-xlm-r-multilingual-v1', 'msbert')
	kcelectra = ("beomi/KcELECTRA-base", 'electra')
	kosimcse = ("BM-K/KoSimCSE-roberta", 'simcse')
	koroberta = ("klue/roberta-base", 'roberta')

	# for ckpt, folder_name in [ko_sbert_multitask, msbert, kcelectra, kosimcse, koroberta]:
	for ckpt, folder_name in [kosimcse]:

		encoder = BertBaseCased(ckpt)
		
		ckpt = ckpt.split('/')[1]
		sentences = index_sentence('ko_sts.csv', f'{folder_name}/word2sent.json', encoder.tokenizer.tokenize)
		encoder.make_hdf5_file(sentences, os.path.join(EMBEDDINGS_PATH, f'{folder_name}/{folder_name}.hdf5'))
