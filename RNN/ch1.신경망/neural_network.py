"""신경망 추론"""

# 입력층, 은닉층, 출력층으로 구성
# 출력 = f(입력값 * 가중치 + 편향)
# h = f(xW + b)
# f는 활성화 함수

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

W1 = np.random.randn(2,4)   # 가중치
b1 = np.random.randn(4)     # 편향
x = np.random.randn(10, 2)  # 입력 - 2차원 데이터 10개가 미니배치로 처리
h = np.matmul(x, W1) + b1   

print(sigmoid(h))   # 출력
