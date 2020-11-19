from matplotlib import pyplot as plt
from typing import List
from typing import Tuple

import math


# Vector =List[float]
#
# height_weight_age = [70,170,40]
#
# grdaes = [95,80,76,62]
#
# def add(v:Vector, w: Vector)-> Vector:
#     assert len(v) == len(w), "Vectors!!"
#     return [v_i + w_i for v_i, w_i in zip(v,w)]
#     assert add([1,2,3],[4,5,6])==[5,7,9]
#
#
# def substract(v, w):
#     assert len(v) == len(w)
#     return [vi - wi for vi, wi in zip(v,w)]
#     assert substract([5,7,9],[4,5,6])== [1,3,4]
#
#
# def dot(v,w):
#     assert len(v)== len(w)
#     return sum(vi*wi for vi, wi in zip(v, w))
# assert dot([1,2,3],[4,5,6])== 32
#
#
# def substract(v,w):
#     assert len(v) == len(w),"dddd"
#     return [vi - wi for vi, wi in zip(v,w)]
#     assert substract([5,7,9],[4,5,6]) == [1,2,3]
#
#
# def sum_of_squares(v:Vector)->float:
#     return dot(v,v)
#
#
# def magnitude(v:Vector) ->float:
#     return math.sqrt(sum_of_squares(v))
#     assert magnitude([3,4])==5
#
#
# def distance(v, w):
#     return magnitude(substract(v,w))
#
# print(distance(3,4))

# A = [[1,2,3],
#      [4,5,6,],
#      [1,2,3],
#      [7,8,9]]
#
# B = [[1,2],
#      [3,4],
#      [5,6]]
#
#
# def f1(a):
#     r = len(A)
#     c = len(A[0]) if A else 0
#     return r,c
#
# print(f1([]))
#
# def f2(A,i):
#     return A[i]
#
# print(f2(A[1],2))
#
# def f3(a,j):
#     return [Ai[j] for Ai in A]
#
#
# print(f3(A[2],2))
#
#
# def f4(r,c,fn):
#     return [[fn(i,j)
#              for j in range(c)]
#             for i in range(r)]
#
# def f5(n):
#     return f4(n,n, lambda i,j : 1 if i == j else 0)
#
#
#
# for i in f5(5):
#     print(i)

# a = [[1,2],
#     [3,4],
#     [5,6]]
#
# b = [[1,2,3],
#      [4,5,6]]
#
# result = []
# 행렬의 열을 구하는 함수
# def get_column(a, b):
#     return [a_i[b] for a_i in a]
#
# def mul_matrix(a, b):
#     #print(len(a[0]), len(b))
#     #assert len(a[0]) == len(b), "행렬 a, b에 대한 곱을 위해서는 a의 열의 개수와 b의 행의 개수가 같아야 합니다."
#     for a_row in a:
#         result_row = []
#         for j in range(len(b[0])):
#             b_col = get_column(b, j)
#             result_row.append(sum(a_row_v * b_col_v
#                                   for a_row_v, b_col_v
#                                   in zip(a_row, b_col)))
#         result.append(result_row)
# mul_matrix(a, b)
#
#
#
# print('=======')
# for rows in result:
#     print(rows)

# def f1(a,b):
#     return [i[b] for i in a]
#
#
# def f2(a,b):
#     for a_row in a:
#         result_row = []
#         for j in range(len(b[0])):
#             b_col = f1(b,j)
#             result_row.append(sum(a_row_v * b_col_v
#                                   for a_row_v, b_col_v
#                                   in zip(a_row, b_col)))
#         result.append(result_row)
# f2(a,b)
#
#
# for v1 in result:
#     print(v1)
# print(f3(A,B))

# import numpy as np
# def f2(A, B):
#     return (np.matrix(A)*np.matrix(B)).tolist()
#
#
# print("결과 : {}".format(f2(a,b)));
#
#
# for i in f2(a,b):
#     print(i)


#------------------------------------------------\
# from collections import Counter
# num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,
#                10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
#                9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,
#                6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,
#                4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
#                2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#
# friends_count = Counter(num_friends)
# xs =  range(101)
# ys = [friends_count[x] for x in xs]
# plt.bar(xs, ys)
# plt.axis([0,101,0,25])
# plt.title("Histogram of Friends Counts")
# plt.xlabel("# of friends")
# plt.ylabel("# of people")
# # plt.show()
#
# num_points = len(num_friends)
# print(num_points)
#
# largest_values=max(num_friends)
# print(largest_values)
# smallest_valeus=min(num_friends)
# print(smallest_valeus)
#
# print(sorted(num_friends))
# a = sorted(num_friends)
# print(a[0])

# def mean(xs):
#     return sum(xs)/len(xs)
#
#
# print(mean(num_friends))
#
# def median_odd(xs):
#     return sorted(xs)[len(xs)//2]
#
# print(median_odd(xs))
#
#
# def median_even(xs):
#     sorted_xs = sorted(xs)
#     hi_midpoint = len(xs)//2
#     return (sorted_xs[hi_midpoint-1]+sorted_xs[hi_midpoint])/2
#
# print(median_even(xs))
#
# def median(v):
#     return median_even(v) if len(v)%2==0 else median_odd(v)
#
# print(median(num_friends))

# def f1():
#     return 10
#
#
# def f2():
#     return f1()
#
# def f3():
#     return f2()
#
#
# print(f3())

#내장함수를 변수로 사용해서 함수가 기능을 상실해버림..
# a = abs(-3)
# print(a)
#
# abs = 10
#
# a = abs(-3)
# print(a)

# def quantile(xs,p):
#     p_index = int(p*len(xs))
#     return sorted(xs)[p_index]
#
#
# print(quantile(num_friends,0.90))
#
#
# def mode(x):
#     count =  Counter(x)
#     max_count = max(count.values())
#     return [x_i for x_i ,count in count.items()
#             if count == max_count]
#     assert set(num_friends) == {1,6}
#
# print(mode(num_friends))
#
# a = [1,2,3,4,5] #<-- 변위라 합니다.
# print("---------------------------------")
# b = mean(a) #<-- 평균이라고 하지요.
# print(b)
# print("---------------------------------")

# for v in a:
#     print(int(v-b),end=',')
#
#comprehension 문장안에 for in은 산술연산이  for 앞으로 가야함
# c = { i - b for i in a}
# print(c)
# print("---------------------------------")
# #편차 절대값의 합의 평균:평균 편차
# d = [abs(i) for i in c]
# print(sum(d)/len(c))
# print("---------------------------------")

#편자 제곱 합의 평균:분산
# f = sum([abs(i)*abs(i) for i in c])/len(c)
# print(f)
# # from scratch.linear_algebra import sum_of_squares
# import math
#
# def de_mean(xs):
#     x_bar = mean(xs)
#     return [x_bar for x in xs]
#
# def variance(xs):
#     n = len(xs)
#     deviations = de_mean(xs)
#     return sum_of_squares(deviations)/(n-1)
#
# def satandard_deviation(xs):
#     return math.sqrt(variance(xs))
#
#
# #and 없이도 출력을 해줌.. 파이썬의 강점이다.
# print(0<=11<=7)
#
# print("--------------------")
#
#
# from enum import Enum
# import random
#
# class kid(Enum):
#     BOY = 0
#     GIRL = 1

# print(kid.BOY.value)
# print(kid.BOY.name)
# print(random.choice([kid.BOY, kid.GIRL]))
#
# class Color(Enum):
#     red = 0
#     green = 1
#     blue = 2
#
# print(random.choice([Color.red,Color.green,Color.blue]))
#
#
# def random_kid():
#     return random.choice([kid.BOY,kid.GIRL])
# both_girls = 0
# older_girls = 0
# either_girs = 0
# random.seed(0)

# ct0= 0; ct1= 1;ct2= 0
#
# for _ in range(1000):
#     a = random.choice([kid.BOY,kid.GIRL])
#     b = random.choice([kid.GIRL,kid.BOY])
#     if a == kid.BOY:
#         ct0 += 1
#     if a == kid.BOY and b == kid.GIRL:
#         ct1 += 1
#     if a ==kid.BOY or b == kid.GIRL:
#         ct2 += 1
#
# print(ct0,ct1,ct2)
# print(ct1/ct0)
# print(ct1/ct2)


# for _ in range(10000):
count = 0

# for _ in range(100000):
#     a = random.randint(1, 10)
#     b = random.randint(1, 10)
#     while a == b:
#         b = random.randint(1, 10)
#
#     c = random.randint(1, 10)
#     d = random.randint(1, 10)
#     while c == d:
#         d = random.randint(1, 10)
#
#     if (a == d and b == c) or (a == c and b == d):
#         count += 1
#
# print(100000 / count, count)

# SQRT_TWO_PI = math.sqrt(2*math.pi)


# def normal_pdf(x,mu,sigma):
#     return (math.exp(-(x-mu)**2/2/sigma ** 2) / (SQRT_TWO_PI * sigma))
#
#
# print(normal_pdf(1,2,3))



# vowel = ['a','e','i','o','u']
#
# word = input("Provided a word to search for vowels")
# found = []
# for letter in word:
#     if letter not in found:
#         found.append(letter)
# for wolves in found:
#     print(vowel)

# phrase = "Don't panic!"
#
# plist = list(phrase)
# print(plist)
#
# for _ in range(4):
#     plist.pop()
#     print(phrase)
#     print(plist)
#     plist.pop(0)
#
#     plist.extend([plist.pop(),plist.pop()])
#
#
# new_phrase = ''.join(plist)
# print(plist)
# print(new_phrase)
#
#
# a = [1,2,3,4]
# b = a
# b.append(100)
# print(a)
# print(b)
# c = a.copy()
# c.append(20000)
# print(a)
#
#
# a = "The largest pizza in the South Korea is sold in Gang_nam Station!!"
# a_list = list(a)
# print(a_list)
#
# print(a_list[0:3])
# d =''.join(a_list[0:3])
# c = ''.join(a_list[4:14])
# print(c)
#
#
#
# time = [2,4,6,8]
# score = [81,93,91,97]
#
# plt.scatter(time,score)
# plt.plot(a,c)
# plt.title("score result of studying time")
# plt.xlabel("# of studying time")
# plt.ylabel("# of score got")
#
# #plt.show()


#a = (x-x/ax*(y-y/ay)의 합//x-x of avg^2 의 합)#선형회귀


time = [2,4,6,8]
score = [81,93,91,97]

X_values = (2-5)*(81-90)+(4-5)*(93-90)+(6-5)*(91-90)+(8-5)*(97-90)
Y_values = (2-5)**2+(4-5)**2+(6-5)**2+(8-5)**2
print(X_values)
print(Y_values)
print(X_values/Y_values)

print(sum([Y_values]))
print(sum([X_values]))



import numpy as np

nx = np.mean(time)
print("x평균:",nx)
ny = np.mean(score)
print("Y평균:",ny)

T1 = sum([(i-nx)*(j-ny)for i,j in zip(time,score)])
T2 = sum((i - nx) ** 2 for i in zip(time))


print(T1)
print(T2)
print(T1/T2)

# a = T1/T2 # 기울기
a = 3
b = nx*a #y 절편
c = ny-b
print(a)
print("A절편:",a)
print("B절편:",ny-b)


e = [(a * i) + c for i in time]
plt.scatter(time, score)
plt.plot(time,e)
plt.title("score result of studying time")
plt.xlabel("# of studying time")
plt.ylabel("# of score got")
plt.show()

# time = [2,4,6,8]
# score = [81,93,91,97] #실제값


#예측값 - 실제값 제곱의 평균\
#평균제곱 오차 //MSE
ev=[84.5, 88.5, 92.5, 96.5]
score = [81,93,91,97]



aw = sum([(i - j) ** 2 for i, j in zip(ev,score)])/4

print(aw)

# 미분 기울기 구하기

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]
xdata = np.array(x)
ydata = np.array(y)
print(type(xdata), xdata) # 배열이다.
a = 0; b = 0
lr = 0.05 # 확습률
for i in range(1000):
    y = a * xdata + b
    a_dif = -(1 / len(xdata)) * sum(xdata * (ydata-y))
    b_dif = -(1 / len(xdata)) * sum(ydata - y)
    a = a - lr * a_dif
    b = b - lr * b_dif
    print(i, round(a,3), round(b,3))

