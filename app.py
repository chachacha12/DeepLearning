
import os
import tensorflow as tf
import shutil # 파이썬 shutil 라이브러리임. 여기선 이미지파일을 다른 폴더로 이동시키기 위해 사용


# (Tensor데이터 (숫자이미지데이터)직접 전처리하기(255로 나누기..) & 만든 모델에 숫자이미지 데이터 넣고 학습하기)
#(코랩에서 작성된 코드들임.)


"""
-->근데 구글 코랩에선 폴더 직접 "새폴더" 이런거 마우스 우클릭해서 직접 만들면 에러있음(숫자이미지로 된 train_ds같은 거 만들었는데 cat, dog 2개의 
클래스로 분류되어서 만들었는데 3classes 라는 등..)

그래서 코랩에선 폴더 직접 만들지말고 파이썬 문법으로 만들기.

일단 기존에 손수 만든 폴더가 있고, 그 폴더가 빈폴더가 아니면 안지워 질거임. 리눅스 명령어 %rm -rf 폴더명  으로 일단 지우기

"""

os.mkdir('/content/dataset')  #파이썬 문법으로 폴더만들기
os.mkdir('/content/dataset/cat') 
os.mkdir('/content/dataset/dog') 


# 이렇게 하면 i는 cat1.jpg.  cat2.jpg   dog1.jpg... 등의 이미지들이 하나씩 들어갈거임
for i in os.listdir('/content/train/'):
  if 'cat' in i:
    shutil.copyfile('/content/train/' + i, '/content/dataset/cat/'+i)       # shutil.copyfile (어떤파일을, 어떤 경로로) # 파일을 다른 경로로 복사해서 붙여주는 명령
  if 'dog' in i:
    shutil.copyfile('/content/train/' + i, '/content/dataset/dog/'+i) 
    
    
    
    
# 이 함수 쓰면, 이미지파일이 있는 경로값만 넣으면 자동으로 이미지들을 숫자이미지set으로 변경해줌
# (텐서플로우에서 쓰는 image_dataset이라는 특별한 타입으로된 숫자이미지set임.. 그리고 이 데이터셋은 이제 바로 training데이터로 딥러닝모델에 넣을 수 있게됨)
# 근데 이 함수 쓰려면 사전작업있음. 모든 고양이사진과 개사진은 다른 폴더에 각각 분류해서 담아둬야함. 그래서 위의 과정 진행
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/dataset/',     # 바꿔줄 이미지들 있는 경로
    image_size = (64,64),    # 여기서 모든 사진들 전처리도 가능. (모든 이미지 사이즈 64 x 64 픽셀로 만들어줌..128 x 128 등 이미지 크게하면 ram 적으면 오래걸릴수도있음)
    batch_size = 64,         # 이건 이미지 2만장을 한번에 모델에 다 넣지않고, batch숫자만큼을 한번에 넣어서 w값 갱신하고 또 그 만큼 넣어서 w갱신함
                             # 여기까지하면  train_ds에는 ( (x값들), (y값들) ) 이런식으로 tuple 혹은 리스트 자료형에 저장이 될거임. 이 train_ds를 모델에 넣으면 학습끝.

                            # 근데 validation 데이터도 준비해보자. (다른 데이터set으로 중간중간 정확도 어케 나오는지 테스트 한번더 해보기위한 것. 그래서 train_ds에서 20% 정도의 데이터 뽑아서 해볼거임.)
    subset = 'training',    # validation 데이터셋 이름 정해줘야함.
    validation_split = 0.2, # 데이터 20%만큼 쪼개주소
    seed = 1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/dataset/',
    image_size = (64,64),
    batch_size = 64,
    subset = 'validation',
    validation_split = 0.2,
    seed = 1234
)


print(train_ds)   # ( (숫자이미지2만개), (0이나 1로된 정답2만개) )   이런식의 shape일거임                                                                         # 코랩말고 로컬컴퓨터에서 하는경우는 경로의 맨앞 슬래쉬 빼고, 로컬경로 잘 잡아주면됨.

"""
학습할때 너무 느리다면 데이터 전처리를 잘 해주면 좀 빨라질 수 있음.
이미지학습에선 이미지데이터 숫자들은 0~255사이의 값 가짐.이걸 0~1사이의 값 가지도록 모두 바꿔주면 좀 더 연산 빨라질거임.
그래서 최적의 w값도 더 빨리 찾을거임. 논문에도 있는 얘기임. 다들 이런 전처리함.
"""

# train_ds는 ( (이미지데이터), (정답)) 이렇게 되어있으므로
def 전처리함수(i, 정답):
  i = tf.cast(i/255.0, tf.float32) # i는 텐서라 그냥 i = i/255.0 하면 안되고 이렇게 해주기. 그리고 만약을 위해 이렇게 자료의 타입을 지정해줘도됨.
  return i, 정답   # 이렇게하면 나누기해서 변환된 i값과 그대로 같은 값은 정답값이 들어갈거임

# 인풋데이터 0~1사이로 압축하기
train_ds = train_ds.map(전처리함수)   # map()은 train_ds 데이터에 전부 함수를 적용시켜줌
val_ds = train_ds.map(전처리함수)



"""
import matplotlib.pyplot as plt # 숫자로된 이미지를 실제 이미지로 확인차 보기위함


# train_ds는 그냥 리스트 자료형이 아니라 BatchDataset.. 자료형임. 64개씩 들어있는 batch의 모음 
# 이 데이터셋이 정말 숫자이미지로 바뀌었는지 출력해보고자함.
# i값은 64개의 데이터일거고 정답값은 64개의 정답들 일것임 
for i, 정답 in train_ds.take(1):
  print(i)
  print(정답) 
  plt.imshow(i[0].numpy().astype('uint8') )  # i는 64개의 숫자이미지들 일것임. 그중 젤 첫번째 이미지를 볼거임. 
                            # 이건 tensor이기 때문에 tensor를 numpy array로 바꿔주려면 .numpy()를 붙여줘야함.
  plt.show()  #실제 이미지를 출력해줌

"""


# 1. 모델 만들기  - convolution과 pooling 여러번 하면 정확도 좀 더 좋아질수도 있기에 해봄
model = tf.keras.Sequential ( [
    tf.keras.layers.Conv2D(32, (3,3), padding="same", activation = 'relu', input_shape = (64,64,3),), # 이건 칼라사진이기 때문에 3넣어야함  
    tf.keras.layers.MaxPooling2D( (2,2) ),  
    tf.keras.layers.Conv2D(64, (3,3), padding="same", activation = 'relu', ),
    tf.keras.layers.MaxPooling2D( (2,2) ), 
    
    tf.keras.layers.Dropout(0.2),    # overfitting문제 완화를 위한 가장 간단한 방법임. (overfitting문제 - 학습용 데이터를 반복 많이해서 컴퓨터가 답을 외운것. 그래서 학습용데이터에선 정확도 95퍼..이런데 처음보는 val데이터에선 80퍼..이런것)
                                     # 윗 레이어의 노드를 일부제거해줌. 이건 아무곳에나 써도됨 (근데 convol +pooling 있는곳에선 보통 pooling뒤에 써줌). 여기에 쓰면 위의 64개 노드중 20프로를 제거해줌.       
    tf.keras.layers.Conv2D(128, (3,3), padding="same", activation = 'relu', ),
    tf.keras.layers.MaxPooling2D( (2,2) ), 
    
    tf.keras.layers.Flatten(),  # 이거 안해주면 마지막 레이어의 결과값이 2차원으로..행렬로 나올거임..그걸 방지하고 [ 0.1  0.2  ....] 등으로 1차원으로 나오게 해주는 함수임. 즉 행렬을 1차원으로 압축해주는 레이어임.. 결론은 이렇게해서 마지막 레이어 결과값을 잘 디자인해줘야 에러없음.
    tf.keras.layers.Dense(128, activation = "relu"), # relu 쓰는 이유는 이미지는 0~255사이의 정수값이 들어가므로 음수나올일 절대 없으므로 이거 많이들 넣어줌.
    tf.keras.layers.Dropout(0.2),
    # 이건 개인지 고양이인지 구분하는 문제임. 그리고 우린 binary_crossentropy 손실함수를 썻음. 그래서 노드 1개만 두고, 개인지 고양인지 0~1사이의 확률로 예측해줌
    tf.keras.layers.Dense(1, activation = "sigmoid" ), 
    
    # 마지막 레이어는 활성함수 있어도되고 없어도됨. 근데 여기서 넣은 이유는 결과를 0~1사이로 압축해서 보고싶기 때문임. softmax는 결과값 0~1로 압축해줌. 여러 카테고리 중 이 사진은 어떤 카테고리에 속할 확률이 높은지 알고싶을때 등에 사용
    # sigmoid: 결과를 0~1로 압축은 동일함. 근데 binary예측문제에서 자주 사용 (ex. 대학원 붙는다/안붙는다...개다 고양이다...즉 0인지 1인지..) -> 마지막 노드갯수는 무조건 1개                 
])

model.summary()


# 2. 모델 컴파일 - 3가지정도 넣어야함
model.compile( loss = "binary_crossentropy", optimizer = "adam", metrics =['accuracy'])


# 3. 모델 학습
model.fit(train_ds, validation_data = val_ds,  epochs = 5)
#  train_ds에는 ( (이미지들),(정답) ) 이렇게 저장이 되어있음. 그래서 fit할때 trainX, trainY 이렇게 
#  따로 넣지말고 train_ds하나만 넣으면됨. 
# val_ds도 똑같음. 그래서 하나만 넣으면 됨


"""
이렇게 해서 학습 진행하면 되고, epochs을 늘려도 overfitting이 일어나기에, val_accuracy 85프로정도가 최대일거임.

Conv2d 더 추가하거나,,
Dense 더 추가하거나,,  한다고해도 정확도 찔끔만 상승함.

가장 중요한건 데이터의 퀄리티임. 데이터 전처리해주는 등..

1. 데이터의 양을 늘리거나
2. 데이터의 질을 늘리거나

이 2가지가 정확도 결과에서 가장 중요함. 다음 시간에 이 2가지를 잡는법을 배울것임. 


"""