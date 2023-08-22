import time
import numpy as np

#랜덤 데이터 생성
np.random.seed(602)
x = np.random.randn(100)

#실제값: a=2, b=0 
y = 2*x + 0.1*np.random.randn(100)

#rmse loss function 정의 
def rmse( y_true, y_pred)):