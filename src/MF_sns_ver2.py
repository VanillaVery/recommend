import pandas as pd
import psycopg2

conn_string = "host=maderi.cka0sue4nsid.ap-northeast-2.rds.amazonaws.com dbname = maderi_rest user = dmk_datasci_yjy password = yjy0408** port=5432"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()

query="""
SELECT k.*, i.name
FROM(
    SELECT channel_id, sns_type, user_id, own
    FROM sns_rival_channel
    UNION
    SELECT channel_id, sns_type, user_id, own
    FROM sns_user_channel
) as k, sns_channel_info as i
WHERE i.channel_id = k.channel_id
"""

#data=pd.read_csv("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/추천시스템/user_sns_reg_data.csv")
data = pd.read_sql_query(query, con=conn)

def myown(own):
    if 'True' in own:
        return 2
    else:
        return 1
data['own'] = data['own'].astype(str)
data['gudok']=data.apply(lambda x: myown(x['own']),axis=1)

df_user_channel=data.pivot_table('gudok',index='user_id',columns='channel_id').fillna(0)
#


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(R,P,Q,non_zeros):
    error=0
    #두 개의 분해된 행렬 P와 Q.T의 내적으로 예측 R행렬 생성
    full_pred_matrix=np.dot(P,Q.T)

    #실제 R행렬에서 널이 아닌 값의 위치 인덱스 추출해 실제 R행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind=[non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind=[non_zero[1] for non_zero in non_zeros]
    R_non_zeros=R[x_non_zero_ind,y_non_zero_ind]
    full_pred_matrix_non_zeros=full_pred_matrix[x_non_zero_ind,y_non_zero_ind]
    mse=mean_squared_error(R_non_zeros,full_pred_matrix_non_zeros)
    rmse=np.sqrt(mse)

    return rmse


def matrix_factorization(R,K,steps=300,learning_rate=0.1,r_lambda=0.01):
    num_users,num_items=R.shape
    #P와 Q 매트릭스의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력
    np.random.seed(1)
    P=np.random.normal(scale=1./K,size=(num_users,K))
    Q=np.random.normal(scale=1./K,size=(num_items,K))

    prev_rmse=10000
    break_count=0

    #R>0인 행 위치, 열 위치, 값을 non-zeros 리스트 객체에 저장
    non_zeros=[(i,j,R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j]>0]

    #SGD기법으로 P와 Q 매트릭스를 계속 업데이트
    for step in range(steps):
        for i,j,r in non_zeros:
            #실제 값과 예측 값의 차이인 오류값 구함
            eij=r-np.dot(P[i,:],Q[j,:].T)
            #Regularization 을 반영한 SGD 업데이트 공식 적용
            P[i,:]=P[i,:]+learning_rate*(eij * Q[j,:] - r_lambda*P[i,:])
            Q[j,:]=Q[j,:]+learning_rate*(eij * P[i,:] - r_lambda*Q[j,:])
        rmse=get_rmse(R,P,Q,non_zeros)
        if (step % 10)==0:
            print("##iteration step:",step," rmse: ",rmse)

    return P, Q

pd.set_option('display.max_rows',None)
#matrix_factorization()함수를 이용해 행렬 분해
P, Q=matrix_factorization(df_user_channel.values,K=100, steps=300, learning_rate=0.1, r_lambda=0.01)
pred_matrix=np.dot(P,Q.T)

title_pred_matrix=pd.DataFrame(data=pred_matrix,index=df_user_channel.index,
                               columns=df_user_channel.columns)

title_pred_matrix.head()

#함수 정의

def get_unseen_movies(ratings_matrix,userId):
    #userid로 입력받은 사용자의 모든 채널 정보를 추출해 Series로 반환
    #반환된 user_rating은 채널명을 인덱스로 가지는 Series 객체임
    user_rating=ratings_matrix.loc[userId,:]

    #user_rating이 0보다 크면 기존에 구독한 채널임. 대상 인덱스를 추출해 list 객체로 만듬
    already_seen=user_rating[user_rating>0].index.tolist()

    # 모든 채널명을 list 객체로 만듬
    movies_list=ratings_matrix.columns.tolist()

    #list compregension으로 already_seen에 해당하는 영화는 movies_list에서 제외함
    unseen_list = [movie for movie in movies_list if movie not in already_seen]

    return unseen_list

def recomm_movie_by_userid(pred_df,userId,unseen_list,top_n=10):
    # 예측 평점 데이터프레임에서 사용자 id인덱스와 unseen_list로 들어온 채널명 칼럼을 추출해 가장 예측 평점이 높은 순으로 정렬
    recomm_movies=pred_df.loc[userId,unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies

#######################################
#사용자가 구독하지 않은 채널명 추출
unseen_list=get_unseen_movies(df_user_channel,6)

#잠재 요인 협업 필터링으로 채널 추천
recomm_movies=recomm_movie_by_userid(title_pred_matrix,6,unseen_list,top_n=10)

#평점 데이터를 DataFrame으로 생성
recomm_movies=pd.DataFrame(data=recomm_movies.values,index=recomm_movies.index,columns=['pred_score'])
recomm_movies=pd.merge(recomm_movies,data[['channel_id','channel_name','sns_type']].drop_duplicates(),how='inner',on='channel_id')

recomm_movies
###########################################################################
# recomm_df=pd.DataFrame()
#
# a=data['user_id']
# for i in list(dict.fromkeys(a)):
#     unseen_list=get_unseen_movies(df_user_channel,i)
#     recomm_movies = recomm_movie_by_userid(title_pred_matrix, i, unseen_list, top_n=10)
#     recomm_movies = pd.DataFrame(data=recomm_movies.values, index=recomm_movies.index, columns=['pred_score'])
#     recomm_movies['userid']=i
#     recomm_movies = pd.merge(recomm_movies, data[['channel_id', 'channel_name', 'sns_type']].drop_duplicates(),
#                              how='inner', on='channel_id')
#     recomm_df=pd.concat([recomm_df,recomm_movies])
# #
# recomm_df.to_csv("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/추천시스템/output_sns210818.csv",mode='w', encoding='utf-8-sig')


###############################################################################

recomm_df=pd.DataFrame()
a=data['user_id']
for i in list(dict.fromkeys(a)):
    unseen_list=get_unseen_movies(df_user_channel,i)
    recomm_movies = recomm_movie_by_userid(title_pred_matrix, i, unseen_list, top_n=10)
    recomm_movies = pd.DataFrame(data=recomm_movies.values, index=recomm_movies.index, columns=['pred_score'])
    recomm_movies['userid']=i
    recomm_movies = pd.merge(recomm_movies, data[['channel_id', 'channel_name', 'sns_type']].drop_duplicates(),
                                 how='inner', on='channel_id')
    recomm_df = pd.concat([recomm_df, recomm_movies])

recomm_df=recomm_df.groupby("userid")["channel_name"].agg(lambda x:list(x))
true_df=data.groupby("user_id")["channel_name"].agg(lambda x:list(x))

pd.merge(true_df.reset_index(),recomm_df.reset_index(),how='inner',left_on='user_id',right_on='userid').to_csv("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/추천시스템/compare_중복x_sns210819.csv",mode='w', encoding='utf-8-sig')