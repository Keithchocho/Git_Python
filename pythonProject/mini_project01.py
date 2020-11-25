import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
param = '수원'
url = 'https://www.diningcode.com/list.php?query='+param
html = requests.get(url)
soup = BeautifulSoup(html.text, "html.parser")
restaurants = soup.find_all("span", attrs={"class": "btxt"})
food = soup.find_all("span", attrs={"class": "stxt"})
score = soup.find_all("span", attrs={"class": "point"})
# location = soup.find_all("i",attrs={"class":"loca"})
location_element = soup.select("span.ctxt")

location = [i.text for i in location_element]
b = location[1::2]
real_add = []
# print(b)
for i in range(len(b)):
    a = b[i].split(" ", 1)
    real_add.append(a[1:2:])


# print(real_add)


for line1, line2, line3, loc2 in zip(restaurants[:], food[:], score[:],real_add[:]):
    print(line1.get_text(), end=' ')
    print(line2.get_text(), end=' : ')
    print(loc2)



