import numpy as np

"""행렬 선언"""

# 하나의 배열에 한가지 자료형을 쓸수 있기 때문에 배열안 모든 데이터들이 통일
# np.array()를 통해 벡터와 행렬 생성
# 다차원 배열 클래스인 np.ndarray 클래스를 생성
# 리스트나 배열을 하나의 인수로 받아야 함
x = np.array([1,2,3])
y = np.array([[1,2,3], [4,5,6]])

#클래스 확인
print(x.__class__)

# 행렬
print('x =', x.shape)
print('y =', y.shape)

# 차원수
print('x.dim =', x.ndim)
print('y.dim =', y.ndim)


"""행렬 연산"""

# 행렬의 연산
# 서로 대응하는 원소끼리 연산
z = x+y
w = x*y

print(z)
print(w)

"""브로드 캐스트"""

# 형상이 다른 배열 끼리의 연산
# 배열의 확장
a = np.array([[1,2], [3,4]])
b = np.array([10, 20])

print(a*10)
print(a*b)

"""벡터의 내적과 행렬의 곱"""

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array([[1,2], [3,4]])
d = np.array([[5,6], [7,8]])

# 벡터의 내적
# 두 벡터에서 대응하는 원소들의 곱을 모두 더한 것
print(np.dot(a,b))



# 행렬의 곱
# 왼쪽 행렬의 행벡터(가로)와 오른쪽 행렬의 열벡터(세로)의 내적
print(np.matmul(c,d))


"""형상 확인"""

# 다차원 배열의 각 차원이 가지고 있는 요소의 개수를 알려주는 작업
# 신경망 구현을 편하게 하기 위해 필요
# a.shape = (m, n)
# b.shape = (p, q)
# 조건: n == p
# a*b = c가 (m,q)배열이 됨

a = np.array([[1,2], [3,4], [5,6]])
b = np.array([[1,1,1,1], [1,1,1,1]])

print('a.shape =', a.shape)
print('b.shape =', b.shape)

c = np.dot(a, b)

print('c.shape =',c.shape)