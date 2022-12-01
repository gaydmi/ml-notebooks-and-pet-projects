import faiss
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import InputExample

class KNN_Predictor():
  """
    Pipeline for sentences' classification based on a KNN algorithm 
    used on a 20 Newsgroup dataset vectorized with sBERT (uses ANN for faster search of the nearest sentences)
  """
  def __init__(self, model: SentenceTransformer, examples: List[InputExample] = None, 
               embedding_size: int = 768, n_clusters: int = 1024):
    """
        Constructs the predictor. 
        Vectorizes the dataset using provided model and indexes it using FAISS ANN library
        :param model
            the Sentence BERT model for the conversion
        :param examples:
            the input examples for the training. If None, Predictor will download the dataset 
        :param embedding_size:
            size of embedding vectors
        :param n_clusters:
            Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) 
            - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    """
    self.embedding_size = embedding_size
    self.n_clusters = n_clusters
    self.corpus_embeddings = []
    self.labels = []
    self.index = None
    self.model = model
    self.get_all_sentence_label_pairs(examples)
    self.index_all_embeddings()
    print('Predictor constructed')


  def get_all_sentence_label_pairs(self, examples):
    if examples == None:
      print('No dataset provided')
      print('Dataset downloading')
      newsgroups_all = fetch_20newsgroups(subset='all', 
                                          remove=('headers', 'footers', 'quotes'), 
                                          return_X_y=True)
      sentences = []
      for ex_index, example in enumerate(tqdm(zip(newsgroups_all[0], newsgroups_all[1]), desc="Dataset vectorization")):
        texts = example[0]
        label = example[1]
        for text in texts.replace('\n','').replace('\t',' ').split('.'):
          sentences.append(text)
          self.labels.append(label)
    else:
      print('Dataset processing')
      sentences = [x.texts[0] for x in examples]
      self.labels = [x.label for x in examples]
      
    print('')
    print('Construction of embeddings')
    self.corpus_embeddings = self.model.encode(sentences)

  def index_all_embeddings(self):
    print('Indexing of all embeddings')
    quantizer = faiss.IndexFlatIP(self.embedding_size)
    self.index = faiss.IndexIVFFlat(quantizer, self.embedding_size, 
                                    self.n_clusters, faiss.METRIC_INNER_PRODUCT)
    self.corpus_embeddings = self.corpus_embeddings / np.linalg.norm(self.corpus_embeddings, axis=1)[:, None]
    self.index.train(self.corpus_embeddings)
    self.index.add(self.corpus_embeddings)

  def knn(self, question_embedding, k):
    question_embedding = question_embedding / np.linalg.norm(question_embedding)
    question_embedding = np.expand_dims(question_embedding, axis=0)
    print(question_embedding, question_embedding.shape)
    distances, corpus_ids = self.index.search(question_embedding, k)
    k_nearest_labels = []
    for id in corpus_ids[0]:
      k_nearest_labels.append(self.labels[id])
    return Counter(k_nearest_labels).most_common(1)[0][0]
      
  def predict(self, query):
    k = 5
    question_embedding = self.model.encode(query)

    print(query, question_embedding)
    return self.knn(question_embedding, k)