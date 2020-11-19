import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



# exclusive or

# print(False^False)
#
# print(False^True)
#
# print(True^False)
#
# print(True^True)

#두개가 같으면 False 다르면 True
#
# a = np.arange(10) # [0 1 2 3 4 5 6 7 8 9]
#
# b = [range(10)] # [range(0, 10)]
#
# print(a)
# print(b)
#
#
#
# for _ in range(10):
#     c = a * 2
#     print(c)
#
# for _ in range(10):
#     d = d * 2
#     print(d)


# a = np.random.randn(3,4)  # (세로, 가로)
# print(a)
#
# b = a+a
# print(b)
#
#
# c =  a + b
# print(c); print('*'*50)
#
# print(a.shape,a.dtype)
#
#
# a = [1,2,3,4]
# b = np.array(a)
# print(b)
#
# a = [[1,2,3,4],
#      [5,6,6,8]]
# c = np.array(a)
# print(c)
# print(a)
#
# print(c.ndim)
#
# print(np.zeros((3,4,2)))  # 수량, 세로, 가로
#
# a = np.array([[2.,3.],
#              [4.,5.]])
#
#
# c =  1/a
# print(c)

# print(4**0.5)  # 4의 루트값이쥬
#
# a = np.arange(10)
# print(a)
# print(a[5:8])
#
# b = a[5:8]
# print(b)
# b[0] = 100
# print(b)
#
#
# a[7]
# print(a)
# print(a[1])
# print(a[2],[1])
#
# a[[1][2]] = 100
# a[[1,2]]= 200
# print(a)

# a = np.array([[1, 2, 3, 4],
#              [5, 6, 7, 8],
#              [9, 10, 11, 12]])
#
# # print(a[1:3, 2:])
# name = ['bob','jea']
#
#
# a  = ['Tiger','Eagle','Lion']
# print(~(a == 'Tiger'))   # ~부정 연산자
# a= [3,4,5,6,8,2,5,1]
# a[0] = 5
# print(a)

# a = np.arange(32).reshape((8,4))
# print(a)


# a = np.array([-10,0,10])
# print(np.isnan(a))
#
# a = np.array([1,2,3])
# b = np.array([2,3,4])
# print(np.add(a,b))
#
# points = np.arange(-5,5,0.01)
# xs,ys =np.meshgrid(points,points)
# z = np.sqrt(xs **2 + ys **2)
# import matplotlib.pyplot as plt
#
# c = plt.imshow(z, cmap=plt.cm.gray);plt.colorbar()
# print(c)
# plt.title("ddadd")
# plt.show()

# x = np.array([1.1,1.2,1.3,1.4,1.5])
# y = np.array([2.1,2.2,2.3,2.4,2.5])
# c = np.array([True,False,True,True,False])
#
# result = [( x if c else y) for x,y,c in zip(x,y,c)]
# print(result)
# result = np.where(c,x,y)
# print(result)
#
# a = np.random.randn(100)
# c = (a>0).sum()
# print(c)
#
#
# a = np.arange(10)
# np.save('some_array', a)
# print(np.load('some_array.npy'))


# x =  np.array([[3,4,5],
#                [5,6,7,]])
#
# y =  np.array([[2,3],
#               [5,6],
#               [7,5]])
#
# c = np.dot(x,y)
# print(c);print(("*"*60))
#
# d = x.dot(y)
# print(d);print(("*"*60))
#
# z = np.dot(x,np.ones(3))
# print(z);print(("*"*60))
#
#
# a = x @ np.ones(3)
# print(a);print(("*"*60))


from numpy.linalg import inv, qr

# x = np.array([[1,2,3],
#               [4,5,6],
#               [7,8,9]])
#
# print(x.T);print("-"*50)
#
# mat = x.T.dot(x)
# print(inv(mat));print("-"*50)
#
# q,r =qr(mat)
# print(q,r);print("-"*50)
#
# print(r);print("-"*50)
#
# a = np.array([1,2,3])
# print(np.diag(a));print("-"*50)
# c = np.trace(np.diag(a))
# print(c);print("-"*50)
#
# c1 = np.invert(x)
# print(c1);print("-"*50)
#
# a = np.array([[1,2],
#               [3,4]])
#
# b = np.array(([3,4],
#               [6,7]))
#
# x = np.linalg.solve(a,b)
# print(x);print("-"*50)


# x = np.array([[2,3],
#               [3,1]])
#
#
# c = np.array([[5],
#              [2]])
#
# x1 = np.invert(x)
# c1 = np.linalg.solve(x,c)
# print(x1);print("-"*50)
# print(c1);print("-"*50)
#
#
# x = [10,20,20,10,10]
# y = [10,10,20,20,10]
#
# A = np.array([[1,0,50],
#               [0,1,30],
#               [0,0,1]])
#
# import math
# co = math.sin(math.radians(30))
# si = math.cos(math.radians(30))
#
#
# Aa = ([[co,-si,0],
#        [si,co,0],
#        [0,0,1]])
#
# # (0,0)원점에서 움직입니다.
# x2 = []
# y2 = []
#
# x1 = [-5,5,5,-5,-5]
# y1 = [-5,-5,5,5,-5]

# 이동합니다.
# for z,k in zip(x1,y1):
#     A1 = np.array([[z], [k], [1]])
#     A3 = np.dot(Aa, A1)
#     x2.append(int(A3[0]))
#     y2.append(int(A3[1]))

# print(x2,y2);print("="*20)
x1 = []
y1 = []

# 처음 꺼에요.
# for i,j in zip(x,y):
#     A1 = np.array([[i],[j],[1]])
#     A2 = np.dot(A,A1)
#     x1.append(int(A2[0]))
#     y1.append(int(A2[1]))

# 회전할게요.

    # x3=[]
    # y3=[]
# def rotation(degree):
#     co = math.sin(math.radians(degree))
#     si = math.cos(math.radians(degree))
#     A5 = ([[co, -si, 0],
#            [si, co, 0],
#            [0, 0, 1]])
#     for z, k in zip(x, y):
#         bv = np.array([[z], [k], [1]])
#         A4 = np.dot(A5, bv)
#         x3.append(int(A4[0]))
#         y3.append(int(A4[1]))

# 확장과 수축
#
# sx1 = []
# sy1 = []
#
# def scale(a,b,sx,sy):
#     AS = np.array([[sx, 0, 0],
#                    [0, sy, 0],
#                    [0, 0, 1]])
#     for z,k in zip(x,y):
#         A1 = np.array([[z], [k], [1]])
#         A3 = np.dot(AS, A1)
#         sx1.append(int(A3[0]))
#         sy1.append(int(A3[1]))
# scale(x,y,3,3)
#
# plt.plot(sx1,sy1)
# plt.scatter(sx1,sy1)
# plt.plot(x3,y3)
# plt.scatter(x3,y3)
# plt.plot(x,y)
# plt.scatter(x,y)
# plt.plot(x1,y1)
# plt.scatter(x1,y1)
# plt.plot(x2,y2)
# plt.scatter(x2,y2)
# plt.show()








# import math
# # plt.scatter()
# plt.plot()
# plt.show()
#
# print(math.sin(math.radians(30)))
# print(math.cos(math.radians(30)))

# import random
# # 랜덤 그래프
# position = 0
# walk = [position]
# steps = 500
#
# for i in range(steps):
#     step = 1 if random.randint(0,1) else -1
#     position += step
#     walk.append(position)
#
# plt.plot(walk)
# plt.show()

# pandas series 연습해보기
import pandas as pd
from pandas import Series, DataFrame

# a = pd.Series([4,7,-5,3])
# print(a)
# print(a.values)


# a1 = pd.Series([4,7,-5,3],index=['d','b','c','z'])
#
# # print(a1)
# # print(a1[0])
# print([a1[a1>0]])
#
# world = {'Hello':2000,'world':3000,'Tiger':2041}
# a2 = pd.Series(world)
# print(a2)
# print(pd.isnull(a2))
#
#
# Data = pd.read_csv('Number_of_death.csv')
#
# print(pd.DataFrame(Data).head(10))
# print(Data['Country/Region'].head(10))
# #print(Data.columns);print("*"*50)\
# Data2 = Data['Country/Region'].head(10)
# Data2['Money_spent_for_cure'] = np.arange(6.)
# print(Data2)


from numpy import nan as NA

# data = pd.Series([1,NA,3.5,NA,7])  # 누락된 정보는 빼고 출력해 줍니다.
# print(data.dropna())
# print(data[data.notnull()])

# data = pd.DataFrame([[1,2,3],
#                      [1,NA,NA],
#                      [NA,NA,NA],
#                      [NA,6,4]])
#
# cleaned = data.dropna(how='all')
# print(cleaned)
#
# data[4]=NA
# print(data.dropna(axis=1,how='all'))
import random


df = pd.DataFrame(np.random.randn(7,3))
df.iloc[:4,1] = NA
df.iloc[:2,2] = NA
# print(df.dropna(thresh=2)) # 몇개 있는 정보의 값만 보고 싶을때 사용 ex) NA값이 하나빠진곳은 출력해줌


# print(df.fillna(3))  #누락된 곳에 값을 넣어줌
# print(df.fillna({1:0.5,2:3}))

data = pd.DataFrame({'k1':['one','Two']*3 + ['Two'], 'k2' : [1,1,2,3,3,4,4]})
print(data.drop_duplicates()) #False값을 DataFrame에 반환