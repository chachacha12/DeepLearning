import tensorflow as tf
import pandas as pd

"""
 - 딥러닝모델에 넣을 CSV 데이터 전처리 하기 (pandas)

"""
data = pd.read_csv('gpascore.csv')    #csv파일을 열어서 읽고 가져온다는뜻. 경로값 넣어주면됨. 근데 이 파일과 같은경로니까 파일이름만 넣음
print(data)  # csv파일의 값들이 출력됨

# 엑셀파일의 데이터 전처리하기 - > 엑셀데이터 행을 보다보면 빵구나서 데이터 없는 공간들도 있음.. 이런 빈부분들은
# 1. 평균값을 넣거나
# 2. 행을 아예 삭제하거나
# 둘중의 한가지를 해야함. 이걸 전처리라고 하는듯 

# 엑셀에 데이터 적으면 직접 처리해주면 되지만, 데이터 많아지면 힘듬. 이걸 판다스로 쉽게 가능.

print( data.isnull().sum() )   # 무슨 열에 빈칸이 몇개인지 세주는 명령임. 

data = data.dropna()  # dropna() 함수쓰면 NaN/빈값있는 행을 제거해줌
# data.fillna(100)  # 이건 빈칸을 채워줌. 평균값 등을 모든 빈칸에 넣어주고 싶을때 사용하기.

# 판다스에 이런 유용한 함수들 있음. 이것만 따로 공부할 필욘없고, 필요할때마다 구글에서 찾아쓰기 

"""
- 유용한 pandas 사용법들

print( data['gre'] )  이렇게 하면 gre열을 모두 출력해줌

print( data['gre'].min() )  이렇게 하면 gre열의 최솟값을 출력해줌 /   max()쓰면 최댓값

print( data['gre'].count() ) 이 열에 몇개의 데이터 있는지 출력

"""

# 판다스 이용해서 y데이터 만들기(일명 label)

y데이터 = data['admit'].values  # 이렇게하면 admit열에 있는 모든 데이터들을 리스트로 만들어줌. 끝


# 판다스 이용해서 x데이터 만들기


x데이터 = [] 

# 반복문안에 i와 rows 처럼 변수를 2개 놓아도됨. i에는 0,1,2,3...등이 들어감. 즉 행의 번호가 들어가고 rows에는 한 행에 있는 4개의 데이터값이 들어갈거임.
for i,rows in data.iterrows():  #  data.iterrows()도 판다스 함수임.  이 함수를 반복문 돌리면 데이터를 한 행씩 출력해볼수있음
    print( rows['gre'] )   #이렇게 하면 각 행들마다 있는 gre값만 출력

# 우리가 원하는 x데이터 입력값 형태는 [380, 3.21, 3] 등 이런형태임. 그래서 리스트로 만들어 줄거임

for i,rows in data.iterrows():  
    x데이터.append( [ rows['gre'], rows['gpa'], rows['rank']  ]  )   # append()쓰면 해당 값이 추가됨

print(x데이터)  



