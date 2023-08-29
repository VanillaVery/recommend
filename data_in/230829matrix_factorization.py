import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.metrics import ndcg_score, average_precision_score
from sklearn.decomposition import NMF, TruncatedSVD

# 1. 데이터 로드
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx'
df = pd.read_excel(url)
#%%
df['Customer ID'] = df['Customer ID'].astype('category') #유저아이디
df['StockCode'] = df['StockCode'].astype('category') #상품아이디
df = df.rename({"Customer ID":"CustomerID"}, axis=1)
#%%
# 2. 피벗 테이블 만들기 -> 너무 적은 인터렉션을 갖는 유저/아이템은 배제
interaction_counts = df.groupby('CustomerID').StockCode.count()
df = df[df.CustomerID.isin(interaction_counts[interaction_counts > 10].index)]
#너무 적게 산 유저는 배제

item_counts = df.StockCode.value_counts()
df = df[df.StockCode.isin(item_counts[item_counts > 10].index)]
#너무 적게 구매된 상품은 배제

pivot = df.pivot_table(index='CustomerID', columns='StockCode', fill_value=0, aggfunc='size')

# 3. implicit data로 변경 (binary화) #이렇게 하는 이유는? 그냥..샀는지 안 샀는지만 보고싶음
pivot = (pivot > 0).astype(int)

# 4. train/test split -> MF에서는 다른 방식!
# 정답 셋 중간중간에 마스킹을 한다고 함 
test_ratio = 0.2
train = pivot.copy()
test = np.zeros(pivot.shape) #똑같은 사이즈지만 0행렬

user = 1
for user in range(pivot.shape[0]):#유저수
    test_interactions = np.random.choice(pivot.values[user, :].nonzero()[0], 
                                         size=int(test_ratio*np.sum(pivot.values[user, :])),
                                         replace=False)
    # nonzero:요소들 중 0이 아닌 값들의 index 들을 반환 (interaction이 있는 item 인덱스 반환)
    # size : interaction 이 있는 item 수 (71개) *0.2
    # 코드 뜻: interaction이 있는 item index중 20%만 뽑아서 test_interaction 으로 지정
    train.values[user, test_interactions] = 0. # 0으로 마스킹
    test[user, test_interactions] = pivot.values[user, test_interactions]
