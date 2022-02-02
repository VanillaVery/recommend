#wordtovec모델 불러오기
import numpy
from gensim import __version__
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
kovec = Word2Vec.load('C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/추천시스템/유의어추천/ko/ko.bin')

kovec.wv.most_similar("강아지")
#wordtovec은 oov 문제를 해결하지 못함 ,

#fasttext pre-trained 모델 불러오기
from gensim import models
from gensim.models import FastText as FT
print(f"== LOAD fasttext START at {datetime.datetime.now()}")
model = FT.load_fasttext_format('C:/Users/윤유진/Downloads/wiki.ko/wiki.ko.bin')
print(f"== LOAD fasttext   END at {datetime.datetime.now()}")
print(model.most_similar('고양이'))
