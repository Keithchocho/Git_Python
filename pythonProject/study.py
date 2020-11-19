# 예제1 import 3가지 방법(유형)

# import matplotlib.pyplot
# matplotlib.pyplot.plot()
# matplotlib.pyplot.show()

# from matplotlib.pyplot import *
# plot()
# show()

import matplotlib.pyplot as plt
# plt.show

# 예제 3

# plot의 종류
# 1. line plot 점과 점 사이를 이어주는
# 2. bar plot 막대 그래프 형태//bar chart
# 3. histogram 연속분포를 나타내는 막대 그래프 bar랑은 다르다
# 4. scatter 점을 이용한 그래프
# 5.contour plot 등치선 그래프 높낮이를 이용하여 보여주는 그래프
# 6.box plot 주식에서 매수 매도 표기용으로사용되는 plot
# 7.pie chart ㅇㅇ 그 차트

# 예제 4
# 그래프의 파이썬의 설정
# plt.title("Tiger")
# plt.show()# 이러면 보여주겠죠?

# 예제 5
# lsit 설정
# plt.plot([1,4,9,16])  #이렇게 쓰면 Y축 data로 입력하게 됩니다.
# plt.show()

# 예제 6
# plt.plot([1,4,9,12],[10,20,30,40])  #앞이 X 축 뒤가 Y축이 됨.
# plt.show()

# 예제 7 #그래프 안에 격자 넣어주기

# plt.grid(True) #grid 순서는 show나오기 전까진 무관하니깐 크게 신경 안써도 되지만 단, 변경가능성이 있는건 show근처로 가면 좋겠쮸??
# plt.show()     #dafault값은 false입니다.

# 예제 8번 x축 y축 라벨 넣기
# plt.xlabel("Tiger")
# plt.ylabel("Lion")
# plt.show()

# 예제9번 차트를 두개 넣으면 두개가 출력되서 비교하기가 유용하다.
# plt.plot([10,20,30,40],[1,4,9,12])
# plt.plot([10,20,30,40],[2,6,11,13])
# plt.show()

# 예제 10 range 를 이용하여 plot사용하기
# plt.plot([10,20,30,40],range(4))
# plt.plot([10,20,30,40],range(2,6))
# plt.show()

# 예제 11
# plt.plot(range(1,5),'r')  #뒤는 색을 나타냄 c m y k w
# plt.plot(range(2,6),'m')
# plt.plot(range(3,7),'g')
# plt.plot(range(4,8),'c')
# plt.plot(range(6,10),'y')
# plt.plot(range(7,11),'k')
# plt.plot(range(8,12),'w')
# plt.grid(True)
# plt.show()

# 예제 12 그래프의 Mark 설정하기
# a = '.,ov^<>w1234sp*hH+xDd'

# for k, v in enumerate( a ):  # enumerate문자 열로도 사용 가능합니다.
#     print(k,v)
#     plt.plot( range( 1+k, 5+k ), v + '-' )
# plt.show()

# 예제 13 선 스타일
# plt.plot(range(1,5),'-')
# plt.plot(range(2,6),'--') #점섬
# plt.plot(range(3,7),'-.') #선과 점선
# plt.plot(range(4,8),':')#세밀한 점선
# plt.show()
# 선 색 스타일 전부넣 넣고싶은 경우
# plt.plot(range(4,8),'g'':')
# plt.show()

# 예제 14  선 세부설정
# plt.plot(
#     [10, 20, 30, 40],  # x축
#     [1, 4, 9, 16], # y축
#     c="b",    # 선 색깔
#     lw=5,     # 선 굵기
#     ls="--",      # 선 스타일
#     marker="o",       # 마커 종류
#     ms=15,        # 마커 크기
#     mec="g",      # 마커 선 색깔
#     mew=5,        # 마커 선 굵기
#     mfc="r"       # 마커 내부 색깔
# )
# plt.show()

# 예제 15 x lim and y lim x 축과 y 축의범위 설정해주기
# plt.plot([0,10,20,30,40],[10,30,40,50,60])
# plt.plot([0,10,20,30,40],[60,50,40,30,10,])
# plt.xlim(-10,30)
# plt.ylim(-20,70)
# plt.grid(True)
# plt.show()

# 예제 16 x,y축 눈금 설정
# plt.plot(range(0,10))
# plt.xticks(range(0,20))#thick의 유효범위 설정 n-1까지 표기됨
# plt.yticks(range(0,100,10))
# plt.show()

# 예제 17 ticks 조금더 고급지게 표현
# plt.plot()
# plt.xticks([1,2,3,4,],["One day","Two Day","Three Day","Four Day"],rotation = "70") #마지막은 글씨 각도 조절
# plt.yticks([1,2,3,4],["Tiger","Dog","Lion","Cat"])
# plt.show()

# 예제 18
# plt.plot(range(1, 5), label="Tiger") #  2  9  1
# plt.plot(range(2, 6), label="Lion")  #  6     7
# plt.legend(loc=4) # 범례 삽입  위치>>  #  3  8  4

# 예제 19 배경 색 바꾸기
# plt.plot()
# plt.gca().set_facecolor([0.2,0.5,1.0]) # 소수점 자리로 바꾸면 됩니다. 250 넘으면 error 발생
# plt.show()

# 예제 20 창의 크기 조절하기
# plt.plot()
# plt.figure(figsize=(6.4*2,4*2))  # 6.4, 4는 default 값
# plt.show()

# 예제 21 한글 입력하기
# import matplotlib.font_manager as fm
# font_location = 'C:/Windows/Fonts/malgun.ttf'                       # 위치값) 오른쪽 마우스 정보 얻음
# font_family = fm.FontProperties(fname=font_location).get_name()
# plt.rc('font', family=font_family)
# plt.plot(['호랑이', '코끼리', '독수리'])
# plt.show()

# 예제 22 서브 플롯

# plot 그래프를 여러개 나타내는것 subplot(plot 가로값, 세로값, 창 순번의 위치값)
# plt.subplot(2, 2, 2)
# plt.plot([10, 20, 30, 40])
# plt.subplot(2, 2, 3)
# plt.plot([10, 20, 30, 40])
# plt.subplot(4, 3, 1)
# plt.plot([10, 20, 30, 40])
# plt.show()

# 예제 23 y축 2개 사용하기
# age = [10,20,30,40,50,60]
# weight = [20, 40, 55, 50, 70, 63]
# height = [100, 120, 140, 150, 170, 165]
# plt.plot(age, weight, 'r')
# plt.twinx()
# plt.plot(age, height, 'g')
# plt.show()

# 예제 24 plot table을 그림판으로 저장함..

# plt.plot(range(1,4))
# a = plt.gcf()
# plt.show()
# a.savefig('hi.png')
#
# plt.plot(range(1,4))
# fig = plt.gcf()
# plt.show()
# fig.savefig('Hello.png')

# 예제 25 bar chart
# plt.bar([1,2,3],[2,3,1])
# plt.show()

# plt.barh([1,2,3],[2,3,1])
# plt.show()

# a = ["python","C++","JAVA","Scala","Leaf","perl","Java script"]
# b = [10,12,13,15,16,20,22]
# plt.subplot(2,2,2)
# plt.bar(a,b,align="center",label='language')
# plt.legend(loc=2)           # edge는 눈금이 모서리에 표시 center는 그래프 정 가운데에 표시 됩니다.
# plt.xticks(rotation="70")   #center =  default
# plt.subplot(2,2,1)
# plt.bar(a,b,align="edge")
# plt.xticks(rotation="70")
# plt.show()

# plt.bar(a,b,width=0.9,bottom=100,align="edge",alpha=0.5)   # alpha = 그래프 투명도  그래프와 배경을 섞는걸 Blending 이라고 함
# plt.show()  # 마우스 우클릭 go to >> declaration and Usages  >> ctrl+B  정보 보기 단축키도 알면 좋겠쮸?


#numpy 사용하는 이유들~~
import numpy as np
# a = np.arange(4)
# a= a+3 # 이게 되네?
# print(type(a))  # 배열타입.. 결과에 ,가 없음
# a = a + 0.35
# print(a)

# 예제 25 그래프 두개 겹치게 해서 비교해보기
# a = plt.bar(np.arange(4) + 0.0, [90, 55, 40, 65], 0.4,label="Tiger",color='r')
# b = plt.bar(np.arange(4) + 0.2, [65, 40, 55, 95], 0.2,label="Lion",color='g')
# plt.legend(loc=9)                 # 맨 뒤 자리는 그래프의 겹침비율을 말하는거 같아요,
# plt.show()

#예제 26
# 그래프 2개를 위 아래로 이어서 출력 비교 하는 방법
# a = plt.bar(np.arange(4), [90, 55, 40, 65], 0.4)
# b = plt.bar(np.arange(4), [65, 40, 55, 95], 0.4,bottom=[90,55,40,65])
# plt.show()


# 연습량
# plt.figure(figsize=(8.4,7))
# num_of_coal =  [150,180,205,234,400,560,811,605,504,430,130]
# year = ([1970,1975,1980,1985,1990,1995,2000,2005,2010,2015,2020])
# plt.bar(year,num_of_coal,width=2.5,color="b",label="#coal",alpha=0.4)
# plt.xlabel("Year by 5years")
# plt.ylabel("Amount * 100_Ton")
# plt.xticks(rotation="70")
# plt.legend(loc=2)
# plt.grid(True)
# plt.show()


# 도표로 자주 나타내는 것들
# 나라별 급여
# 직업별 월급치아ㅣ
# 성별 급여차이
# 학과별 경쟁률
# 월별 수익률
# 팀별 성적
# 동물원 특정객체의 변화
# 종교 유무에 따른 이혼율
#
# 성별 직업에 대한 빈곤수
# 지역별 연령대비율
# 연령대별 정당의 대한 지지율
# 지역별 정당의 대한 지지율
# 나라별 금은동에 대한 매달 수
# 도시별 성별에대한 범죄율
#
# 경제 서장률
# 자살율
# 합격률
# 취업률
# 백분률(프로야구 타율)
# 판매량
# 재고량
# 성장률
# 생산량
# 분포량
# 빈도
# 유입량
# 이자율

import pandas as pd

# data 분석을 위한 library
# 1차원 자료는 serries
# 2차원 자료는 Data Frame
# DataFrame 생성
# data ={"이름":["개구리","고길동"," 둘리"],
#        "나이":[23,45,32],
#        "고향":["서울시"," 인천"," 분당"]}
#
# DF = pd.DataFrame(data)
# print(DF)

# a = [10,20,30,40]   # 많은 부분에서, field 의 성격을 지닌다.
# b = [50,60,70,80]
# c = [a,b]
#
#
# DF = pd.DataFrame(c)   # .T 붙이면 세로로 입력되고 안붙이면 가로로 데이터 들어감
# DF.columns=['a','b','c','d'] # 행에 타이틀 넣어주기
# print(DF)


#CRUD의 C라고 보면 됩니다.
DF = pd.DataFrame(columns=("이름","나이","고향"))
# print(len(DF))
DF.loc[20] = ["Tiger",18,"Seoul"]
DF.loc[30] = ["Eagle",50,"Suwon"]
DF.loc[30] = ["Hippo",20,"Suwon"]
for i in range(10):
    DF.loc[len(DF)]=["이순신" + str(i), 10+i, "고향" + str(i)]

# loc[20]은 키값  만약 key value 가 같으면 갱신됨
# print(DF)
# print(len(DF))
#
# #Data 갱신
# DF.loc[30] = ["Eagle",50,"Suwon"]
# DF.loc[30] = ["Hippo",20,"Suwon"]
# print("-"*30)
#
# #Data 삭제
# c = DF.drop([30, 20, 3, 4, 5])
# print(c)

print(DF.loc[6])   # 내가 찾고자 하는 정보를 찾을떄 하는 방법
print(DF[0:4])     # 0부터 시작해서 4전까지
print("--"*30)
print(DF)
print(DF.head())  # 상단 5개 정도의 Sampple Data를 출력시켜줌
print(DF.head(3)) # 괄호안에 숫자를 입력하면 내가 입력한 만큼 보여줌
print(DF.tail());print("-"*30)  # 반대로 하단의 Sample을 출력시켜주고 사용 방법은 HEAD랑 동일함

