"""loss function, stochastic gradient descent방식을 비교하는 코드"""
import time
import numpy as np

#랜덤 데이터 생성
np.random.seed(602)
x = np.random.randn(100) #노멀분포에서 100개생성

#실제값, a=2, b=0 을 찾도록 유도될 것임
y = 2*x + 0.1*np.random.randn(100)

#rmse loss function 정의 
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

#%%
"""brute-force 방법(무작위 대입법)"""
start_time = time.time()
#-10~10 사이의 값을 1000개의 interval로 쪼갬(이 안에 값이 있을거다...)
a_values = np.linspace(-10,10,1000)
b_values = np.linspace(-10,10,1000)

min_loss = float('inf') # 로스의 값을 처음에 가장 큰 값으로 초기화 해줌
best_a, best_b = None, None

for a in a_values:
    for b in b_values:

        y_pred = a*x + b
        loss = rmse(y, y_pred)

        #최소 로스를 기록해서, 그 보다 낮은 값이 등장할 경우 로스와 최적의 파라미터를 update
        if loss < min_loss:
            min_loss = loss
            best_a, best_b = a, b

brute_force_time = time.time() - start_time
print(f'brute_force_method: a = {best_a},b = {best_b},rmse={min_loss},Time={brute_force_time}seconds')
#brute_force_method: a = 1.9919919919919913,b = -0.010010010010010006,rmse=0.09445721999361457,Time=32.28336715698242seconds
#%%
"""sgd:하나의 샘플만 뽑아서 gd를 진행"""
start_time = time.time()

#무작위로 a, b 선정 (좀 더 합리적)
a, b = np.random.randn(),np.random.randn()

# 임의의 학습률 
learning_rate = 0.1

#1000번의 에포크에 걸쳐 업데이트 진행 
for epoch in range(1000):
    random_idx = np.random.choice(len(x)) # sgd는 step마다 하나의 랜덤 샘플만을 사용 
    xi,yi = x[random_idx], y[random_idx]

    #예측값 생성
    y_pred = a*xi + b

    #mse의 도함수 (rmse와 최적값이 동일)
    gradient_a = -2*xi*(yi - y_pred)
    gradient_b = -2*(yi - y_pred)

    a = a - learning_rate*gradient_a
    b = b - learning_rate*gradient_b

    #각 100번째 스탭마다 parameter와 loss값 계산
    if epoch % 100 == 0:
        y_pred = a*x + b
        loss = rmse(y,y_pred)
        print(f'ecpoch {epoch}, a={a}, b={b}, RMSE={loss}')

#소요시간확인
sgd_time = time.time() - start_time
print(f'sgd method: a={a},b={b},rmse={loss},time={sgd_time}seconds')
#%%

