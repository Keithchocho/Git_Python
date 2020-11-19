import pandas as pd
import cx_Oracle as co
# lsnrctl status 를 사용하여 sql 환경설정(Host, port, user)
# 첫번째 방법

dsn_tns = co.makedsn("localhost", "1521",'orcl')
conn = co.connect('cdh','1234',dsn_tns)
cursor = conn.cursor()
cursor.execute("""SELECT * FROM tab01""")  #<- 주석으로 명령어 입력
# 출력의 4가지 방법
# print(cursor.fetchone())
print(cursor.fetchall())
# for i in cursor:
#     print(i)
# df = pd.DataFrame(cursor)
# print(df)