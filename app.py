import tensorflow as tf
import matplotlib.pyplot as plt  # 이미지를 파이썬으로 띄워서 볼 수 있는 라이브러리. 터미널에 pip install matplotlib 해서 설치해줬음.
import numpy as np  # 이 라이브러리로 numpy array자료의 shape 변경시킬 수 있음


#(trainX, trainY ), (testX, testY)이건...  numpy array자료임. numpy array자료형이나 list나 tensor나 다 비슷비슷한 애들
(trainX, trainY ), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()  #텐서플로우에서 제공해주는 몇가지 데이터셋 중 하나 

# 이미지 이용한 딥러닝할때, 이미지 전처리해줄때, 보통은 원래 0 - 255사이의 값 저장되어있는 값 바로 넣지않고 0- 1로 압축해서 넣어줌 . 
# 이렇게 미리 압축하고 넣으면 결과가 좋을수도있고, 처리시간이 빨라질수도 있기때문.
# 그래서 255로 데이터를 나누면 0 - 1 사이로 압축될거임.  이 전처리는 선택사항임. 이렇게 해보고 결과 더 잘나오면 이렇게 하면되고..
trainX = trainX / 255.0
testX = testX / 255.0



#convolution layer 적용하기위해 데이터들 깔끔하게 전처리하기
trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1) )  # numpy 라이브러리를 통해 데이터의 shape를 바꿈,, 3차원에서 4차원으로 변경함. 모델에 적용하는 convolutional layer의 Conv2D() 이용하려면 4차원 데이터 있어야함.
testX = testX.reshape( (testX.shape[0], 28, 28, 1) )   # trainX.shpae가 (60000,28,28)이라 0번째 인덱스는 60000이 들어갈거임 
                                               # 이건 흑백사진이라 그렇고, 칼라사진이면 픽셀하나에 3개의 rgb데이터가 들어가 있음..그래서 (60000, 28,28,3)이렇게 되어야할거임.
                                            
        
# 1. 모델 만들기 
model = tf.keras.Sequential ( [
 
    #   1번. 첫번째 레이어에 convolution layer 적용하기
    tf.keras.layers.Conv2D(32, (3,3), padding="same", activation = 'relu', input_shape = (28,28,1),  ), 
    # 첫인자는 서로다른 32개 커널로 feature를 추출해서 복사본 32개를 만들어주세요임.  3,3은 커널수임. 기본적으로 3,3부터 시작하면되고. 실험적으로 결과잘나올동안 변경해주면됨.
    # convolution 적용하면 커널 거쳐서 이미지가 압축돼서 조금씩은 작아짐..(가로세로1픽셀 정도..) 그래서 (28,28)이었던 이미지가 (27,27)이렇게됨.. 그걸 방지해주고자 padding씀. 이거쓰면 바깥쪽에 부족한 1픽셀씩 여백으로 그냥 더해주는 녀석인듯. 이미지 사이즈 맞추고 싶을떄 씀.
    # 왜 relu함수씀?:  relu 액티베이션 함수는 음수의 값은 다 0으로 바꿔줌. 이미지를 숫자로 바꾸면 0 - 255사이이므로. 이미지는 음수데이터 있을수없기에 그냥 relu 자주씀
    # input_shape 쓰는 이유는 일단 뒤에서 summary()함수 쓰려면 필요하기도하고, 안쓰면 ndim(number of dimention)에러 발생할수있음.  ndim에러 되게 흔한 에러인데
    # conv2D 레이어는 4차원의 데이터를 필요로함. 4차원데이터는 (60000, 28,28,1) 이런식으로 4개가 구성되어 있어야함. 근데 우리가 여기서 넣은 데이터는 (60000,28,28)이었을거임. 
    # 그래서 하나의 데이터 모양넣는 input_shape을 3차원으로 변경해주면됨. (28,28,1)은 그냥 1개의 리스트데이터가 28개 있고, 그게 또 28개 있다는 뜻임.
    
    #  2번. 이미지의 중요한 포인트를 가운데로 모아주기
    tf.keras.layers.MaxPooling2D( (2,2) ),  
    # 이미지사이즈를 줄여주고 중요한 특징들을 모아줌, (2,2)는 pooling사이즈임.. 즉 기존 이미지값을 (2,2)픽셀씩 묶어서 하나의 픽셀?로 압축시켜준거임 

    # tf.keras.layers.Dense(128, input_shape = (28,28),  activation = "relu"), # relu 활성함수는 음수값은 다 0으로 만들어주는 녀석임. 나중에 배우는 convolution layer에서 자주씀.
    tf.keras.layers.Flatten(),  # 이거 안해주면 마지막 레이어의 결과값이 2차원으로..행렬로 나올거임..그걸 방지하고 [ 0.1  0.2  ....] 등으로 1차원으로 나오게 해주는 함수임. 즉 행렬을 1차원으로 압축해주는 레이어임.. 결론은 이렇게해서 마지막 레이어 결과값을 잘 디자인해줘야 에러없음.
    tf.keras.layers.Dense(64, activation = "relu"), # relu 쓰는 이유는 이미지는 0~255사이의 정수값이 들어가므로 음수나올일 절대 없으므로 이거 많이들 넣어줌.
    tf.keras.layers.Dense(10, activation = "softmax" ), # 마지막 레이어는 활성함수 있어도되고 없어도됨. 근데 여기서 넣은 이유는 결과를 0~1사이로 압축해서 보고싶기 때문임. softmax는 결과값 0~1로 압축해줌. 여러 카테고리 중 이 사진은 어떤 카테고리에 속할 확률이 높은지 알고싶을때 등에 사용
                                                        # sigmoid: 결과를 0~1로 압축은 동일함. 근데 binary예측문제에서 자주 사용 (ex. 대학원 붙는다/안붙는다...개다 고양이다...즉 0인지 1인지..) -> 마지막 노드갯수는 무조건 1개                   
   
])

"""
컨볼루션포함한 모델의 레이어 구성순서 

conv - pooling 두줄 계속 복붙해서 인자값 변경해가며 여러번 함.
그리고 Flatten - Dense - 출력  
"""

model.summary()


# 2. 모델 컴파일 - 3가지정도 넣어야함
model.compile( loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics =['accuracy'])


# 3. 모델 학습
model.fit(trainX, trainY, validation_data= (testX, testY), epochs = 1)



# 모델이 제대로 만들어졌는지 학습 후 평가해보기. 
score = model.evaluate(testX, testY)
print(score)  # -> 출력하면   [0.32401221990585327, 0.8826000690460205] 이런식으로 데이터2개 나옴. 첫째는 오차값이고, 두번째는 정확도임. 
# 인자값에 x데이터와 y데이터 작성해줌. 근데 트레이닝에 썻던 데이터들을 쓰면안됨. trainX, trainY 안됨.. 왜냐면 이미 컴퓨터가 학습하면서 많이 봐서 외웠기 때문.
# 첨보는 데이터를 넣어줘야함. 


"""
overfitting 현상

새로운 test데이터로 평가해서 나온 정확도와, 실제 학습해서 마지막으로 도출된 정확도를 비교해보면 학습한 값의 정확도가 좀 더 높은걸 말하는 현상
이유는 인간도 비슷한 수학문제 계속 풀다보면 정답을 외움. 그것처럼 컴퓨터도 정확도 높이려고 기존 답을 외운것뿐임. 그래서 새로운 문제에선 정확도 좀 낮음.
이 문제 해결하기위한 여러가지 방법을 사람들이 도입함.


첫번째 방법: fit()함수안에 validation_data넣기
이렇게 넣으면 epoch 1회 진행될떄마다 validation_loss와 validation_accuracy 같은것도 같이 나옴. 수백번 epoch를 진행하면 overfitting 발생확률 매우 높음. 
그래서 그렇게 되기전에, 학습이 더이상 잘 안 일어날때, 학습을 정지해서 그때의 모델을 뽑을 수 있을거임. 즉 이런식으로 이용가능
val_accuracy를 높일 방법을 더 많이 찾아보자.. 그게 중요함.
-Dense layer 추가?
-Conv+pooling 추가?  등 여러 방법으로 혼자 연구해보면 좋음

"""


