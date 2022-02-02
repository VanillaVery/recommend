import io
import re
import string


import pandas as pd
from tqdm import tqdm
import konlpy
from konlpy.tag import Mecab
import numpy as np
from konlpy.tag import Okt
import tensorflow as tf
from tensorflow.keras import layers
import re
import psycopg2
import os
import itertools
import json
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



conn_string = "host=maderi.cka0sue4nsid.ap-northeast-2.rds.amazonaws.com dbname = maderi_raw user = dmk_datasci_yjy password = yjy0408** port=5432"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query="""
SELECT content FROM t_buzz_contents_20211108 where service='naver_news'
union
SELECT content FROM t_buzz_contents_20211107 where service='naver_news'
union
SELECT content FROM t_buzz_contents_20211106 where service='naver_news'
union
SELECT content FROM t_buzz_contents_20211105 where service='naver_news'
union
SELECT content FROM t_buzz_contents_20211104 where service='naver_news'
union
SELECT content FROM t_buzz_contents_20211103 where service='naver_news'
union
SELECT content FROM t_buzz_contents_20211102 where service='naver_news'
"""
data = pd.read_sql_query(query, con=conn)
print(data.shape)
datalist=[i for i in data['content'] if 20<=len(str(i).split(' '))<=500]#일단 리스트로 가져와보기
datalist = random.sample(datalist, 110000)

#datalist=datalist[:100]#일단 100개만 돌려보za
#%%
def clean_korean(sent):
    if type(sent) == str:
        h = re.compile('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]+')
        result = h.sub(' ', sent)
        result=result.replace('\n','').strip()
    else:
        result = ''
    return result


datalist_clean=[clean_korean(i) for i in datalist]

#%%
m=Mecab()

clean_m_text=[m.pos(x) for x in tqdm(datalist_clean)]
useful_tag = ('NNP','NNG')
clean_m_useful_text = []
for i in tqdm(range(len(clean_m_text))):
    # x=clean_okt_text[0][4]
    sent = [x for x in clean_m_text[i] if (x[1] in useful_tag) & (len(x[0].strip())!=1)]
    clean_m_useful_text.append(sent)

#pos 지우기
preprocessed_text = []

for i in tqdm(range(len(clean_m_useful_text))):
    sent = [x[0] for x in clean_m_useful_text[i]]
    preprocessed_text.append(sent)

preprocessed_text_count=list(itertools.chain(*preprocessed_text))

from collections import Counter
count = Counter(preprocessed_text_count)

tag_count = []
tags = []

for n, c in count.most_common(33000):
    dics = {'tag': n, 'count': c}
    if len(dics['tag']) >= 2 and dics['count']>=10 :
        tag_count.append(dics)
        tags.append(dics['tag'])

print(tag_count)
print(tags)
print(len(tags))

file_tag="../tag_count1.json"

with open(file_tag, 'w') as outfile:
    json.dump(tag_count, outfile)


preprocessed_text=[' '.join([j for j in preprocessed_text[i] if j in tags]) for i in tqdm(range(len(preprocessed_text)))]

print(preprocessed_text[:10])
#%%
from tensorflow.keras import preprocessing
tokenizer=preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(preprocessed_text)
sequences=tokenizer.texts_to_sequences(preprocessed_text)
vocab=tokenizer.word_index

preprocessed_text[0]
vocab_reverse = {i:x for x,i in vocab.items()}
vocab_reverse[364]

vocab['pad'] = 0
vocab_size = len(vocab)

file_path_vocab="../vocab1.json"
file_path_reverse="../vocab_reverse1.json"

with open(file_path_vocab, 'w') as outfile:
    json.dump(vocab, outfile)
with open(file_path_reverse, 'w') as outfile:
    json.dump(vocab_reverse, outfile)


#%%
# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels
#%%
window_size=3
num_ns=3

targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=window_size,
    num_ns=num_ns,
    vocab_size=vocab_size,
    seed=62)
#%%

#%%
targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")
#%%
BATCH_SIZE = 512
BUFFER_SIZE = len(targets)

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
#%%
class Word2Vec(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim):
        super(Word2Vec,self).__init__()
        self.target_embedding=layers.Embedding(vocab_size,
                                               embedding_dim,
                                               input_length=1,
                                               name="w2v_embedding")
        self.context_embedding=layers.Embedding(vocab_size,
                                                embedding_dim,
                                                input_length=num_ns+1)

    def call(self,pair):
        target,context=pair

        if len(target.shape)==2:
            target=tf.squeeze(target,axis=1)
        word_emb=self.target_embedding(target)
        context_emb=self.context_embedding(context)
        dots=tf.einsum('be,bce->bc',word_emb,context_emb)
        return dots
#%%
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])



word2vec.fit(dataset, epochs=20)
word2vec.save("../w2vTF_model1")
#%%
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
norms = np.linalg.norm(weights, axis=1)
#%%
'''maximum simliarity word 찾기'''
topk = 10
idx = vocab['쯔양'] # 찾고자 하는 단어
sim = weights[idx, :] @ weights.T
cosine_sim = sim / (np.linalg.norm(weights[idx, :]) * norms)
topk_words = np.argsort(cosine_sim)[-topk:] # 뒤에서부터 가장 비슷한 단어임!
topk_words_score = [(vocab_reverse.get(i), c) for i, c in zip(topk_words, cosine_sim[topk_words])][::-1]
print(topk_words_score)
#%%
