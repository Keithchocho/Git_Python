import pandas as pd
import folium as fo
from matplotlib import pyplot as plt
from matplotlib import animation as ani
import seaborn as sns
import numpy as np
import folium



# 데이터 불러오기
#
# Data = pd.read_csv('Number_of_death.csv')
#
#
# # 열 이름 출력
# print(Data.columns);print("*"*50)
#
# # 행 확인해보기
# print(Data.head());print("*"*50)
#
# # 사망자 수 정렬해보기
# Deaths_values = Data.sort_values(by = 'Deaths', ascending=False).head(10)
# print(Deaths_values);print("*"*50)
# extract_number = Deaths_values[['Country/Region','Deaths']]
# print(extract_number);print("*"*50)
# sns.barplot(x = 'Country/Region', y = 'Deaths', data=extract_number)
# plt.xticks(rotation='45')
# plt.xlim(0,10)
# plt.show()
# 데이터 분석해보기
# Evaluation_death = Deaths_values.describe()
# print(Evaluation_death)



# Wrold_confirmed = pd.read_csv('C:/Users/BIT-057/Desktop/Data/CONVENIENT_global_confirmed_cases.csv')
# print(Wrold_confirmed)

Seoul_Data = pd.read_csv('PatientInfo.csv').head(5)
print(Seoul_Data)


# Seoul_map = folium.Map(location=[37.532094, 126.994344],zoom_start=13)
# print(Seoul_map)
#
#
# # Excel 불러서 읽기
# xlsx = pd.ExcelFile('ex1.xlsx')
# b = pd.read_excel(xlsx,'Sheet1')
# print(b)


map = folium.Map(location=[37.532094, 126.994344],
                 tiles='Stamen Terrain',)
print(map)


import requests
import json
url = ' http://opendata.kwater.or.kr/openapi-data/service/pubd/supply/watersupplylist'

resp= requests.get(url)
print(resp)

