import tqdm
import json
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import h5py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import keras

pd.set_option('display.max_seq_items', None)

path="C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/문서/GitHub/NLPProjects/recommend/w2vTF_model1"
file_path_vocab = "C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/문서/GitHub/NLPProjects/recommend/vocab_soynlp.json"
file_path_vocab_reverse = "C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/문서/GitHub/NLPProjects/recommend/vocab_reverse1_soynlp.json"
file_path_tag_count = "C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/문서/GitHub/NLPProjects/recommend/tag_count_soynlp.json"

word2vec=tf.keras.models.load_model(path)
with open(file_path_vocab, "r") as json_file:
    vocab = json.load(json_file)

with open(file_path_vocab_reverse, "r") as json_file:
    vocab_reverse = json.load(json_file)

with open(file_path_tag_count, "r") as json_file:
    tag_count = json.load(json_file)

pd.DataFrame(tag_count).to_csv("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/추천시스템/211108w2v_keyword/tag_count_soynlp1.csv",mode='w',encoding='utf-8-sig')

word2vec

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
norms = np.linalg.norm(weights, axis=1)
#%%
'''maximum simliarity word 찾기'''


topk = 10
idx = vocab['아이폰'] # 찾고자 하는 단어
sim = weights[idx, :] @ weights.T
cosine_sim = sim / (np.linalg.norm(weights[idx, :]) * norms)
topk_words = np.argsort(cosine_sim)[-topk:] # 뒤에서부터 가장 비슷한 단어임!
topk_words_score = [(vocab_reverse.get("{}".format(i)), c) for i, c in zip(topk_words, cosine_sim[topk_words])][::-1]
print(topk_words_score)

######################################################################
keys=[]
for j in range(len(tag_count)):
    key=tag_count[j]['tag']
    keys.append(key)
values=[]
for j in tqdm.tqdm(keys):
    topk = 11
    idx = vocab['{}'.format(j.strip())] # 찾고자 하는 단어
    sim = weights[idx, :] @ weights.T
    cosine_sim = sim / (np.linalg.norm(weights[idx, :]) * norms)
    topk_words = np.argsort(cosine_sim)[-topk:] # 뒤에서부터 가장 비슷한 단어임!
    topk_words_score = [(vocab_reverse.get("{}".format(i)), c) for i, c in zip(topk_words, cosine_sim[topk_words])][::-1]
    values.append(topk_words_score)


pd.DataFrame(values).to_csv("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/추천시스템/211108w2v_keyword/values1207_soynlp.csv",mode='w',encoding='utf-8-sig')