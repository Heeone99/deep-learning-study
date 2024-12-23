"""노드와 계층을 구현"""

import numpy as np

"""Repeat 노드"""

# N개로 분기하는 노드
# 입력 데이터를 여러 개의 복사본으로 확장
# 데이터를 배치 단위로 확장하거나, 여러 출력에 동일한 입력을 분배할 때 사용.
# 반복적인 계산에 동일한 입력을 사용하는 구조에서 유용.

D, N = 8, 7
x = np.random.randn(1, D)   # 입력
y = np.repeat(x, N, axis=0) # 순전파
dy = np.random.randn(N, D)  # 무작위 기울기
dx = np.sum(dy, axis=0, keepdims=True)

"""Sum 노드"""

# 입력 데이터를 특정 축에 따라 합산.
# 배치 데이터를 하나로 집계하거나, 특정 방향으로 데이터 차원을 줄이는 경우.
# 가중치를 계산할 때 자주 사용.

D, N = 8, 7
x = np.random.randn(N, D)   # 입력
y = np.sum(x, axis=0, keepdims=True)    # 순전파
dy = np.random.randn(1, D)  # 무작위 기울기
dx = np.repeat(dy, N, axis=0)   # 역전파

"""MatMul 노드"""

# 행렬 곱셈 수행
# 가중치와 입력 데이터를 곱해 신경망의 출력값을 계산.
# 신경망의 기본 구성 요소로, 대부분의 층에서 사용.

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

        
"""완전연결계층"""

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


"""Sigmoid"""

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


"""Softmax"""

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx