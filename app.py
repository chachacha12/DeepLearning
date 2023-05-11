import tensorflow as tf
import pandas as pd
import numpy as np   # 파이썬으로 다차원 리스트, 행렬 만들때 쓰는 라이브러리



# csv파일 가져옴
data = pd.read_csv('gpascore.csv')    #csv파일을 열어서 읽고 가져온다는뜻. 경로값 넣어주면됨. 근데 이 파일과 같은경로니까 파일이름만 넣음
data = data.dropna()  # dropna() 함수쓰면 NaN/빈값있는 행을 제거해줌


# x데이터, y데이터 만듬
y데이터 = data['admit'].values  # 이렇게하면 admit열에 있는 모든 데이터들을 리스트로 만들어줌. 끝
x데이터 = [] 
for i,rows in data.iterrows():  
    x데이터.append( [ rows['gre'], rows['gpa'], rows['rank']  ]  )   # append()쓰면 해당 값이 추가됨


# 만들었던 딥러닝모델
model = tf.keras.models.Sequential([ 
    tf.keras.layers.Dense(64, activation='tanh'),   #첫번째 레이어임. 레이어만드는 공식임. 파라미터값은 그 레이어에 들어가는 노드갯수임. 
    tf.keras.layers.Dense(128, activation='tanh'),   # 노드갯수는 결과 잘 나올때까지 실험적으로 때려박아 파악해야하는거임. 근데 관습적으로 2의제곱수를 많이 넣음 
    tf.keras.layers.Dense(1, activation='sigmoid'),   # 마지막 결과는 0또는 1 등 하나일경우엔 1 적으면됨. 근데 결과를 여러개로 다양하게 만들고싶다면 1이상 놓으면됨 
])

# 모델 컴파일
model.compile(optimizer = 'adam' , loss ='binary_crossentropy' , metrics=['accuracy'])


 
 # 이 코드가 딥러닝학습 시작해라~  즉, w값 찾으라는 동작시킴  
 # //  run했을때 Epoch 갯수에 맞게 뜨면서 출력될거임. 그때 나오는 loss값은 예측값과 실제값의 오차가 얼마인지 나타내는거임. 즉 점점 작아질수록 좋음.
 # //  그리고 accuracy는 예측값이 실제값과 얼마나 맞는지 평가해주는 거임. 즉, 높을수록 좋음. 
 # //  그래서 epochs값을 100이나 1000정도로 올리면 실제로 accuracy가 0.8 정도까지 올라감.  근데 약간 운빨로..학습후에 마지막 정확도 낮게 나올수도. 그럴땐 다시 돌리기. 가장 좋은결과 나올때 저장하면됨. 
model.fit( np.array(x데이터) , np.array(y데이터), epochs=10)

"""
근데 fit함수안에 x데이터와 y데이터를 그냥 파이썬의 리스트타입으로 넣으면 안되고
 -numpy array
 -tf tensor
이 두가지 중 하나의 타입으로 변환을 해서 넣어줘야 실행됨.
그래서 numpy array로 변환해서 넣어줌.  np.array(x데이터) 하면 변환됨
"""


# 이제 fit함수를 실행했으면 모델이 알아서 학습을 epochs횟수만큼함. 그 후 어떻게 하나?


# 예측하기
"""
학습시킨 모델로 예측해보기
    ex) GRE 성적이 700, 학점 3.7, Rank 4 일때 합격확률은..?

"""
예측값 = model.predict(  [[750, 3.70, 3], [400, 2.2, 1] ] )   #만든 model로 y값 예측해주는 함수 /  x데이터값만 넣어주면됨.  예시처럼 여러개 넣어줘도됨
print(예측값)



"""
@@@ 딥러닝의 기초과정 @@@

1. 모델 만들고
2. 데이터 집어넣고 학습
3. 새로운 데이터 예측

이게 앞으로 계속 할 내용임.

근데 여기서 성능향상시키는 방법들 존재함
 - 데이터 전처리 깔끔하게 잘하기
 - 파라미터튜닝 (모델만들때 Denso함수에 넣은 노드갯수 등을 바꾸거나 layers를 하나 더 두거나...바꿔가면서 해보는 것)

---> 딥러닝 학습은 실험이 매우 중요함.. 여러 파라미터값들과 activation함수나 optimizer 등을 바꿔가면서 실험해보며 정확도 올려야하는거임.
정확도가 80프로 정도가 거의 최대치일거임. 

"""