import tensorflow as tf
import pandas as pd

#대학입학 확률예측 프로그램을 텐서플로 써서 간단히 만들어볼거임


tf.keras #keras는 텐서플로우 안에 있는 딥러닝 도와주는 좋은 도구. 딥러닝 훨씬 간단하게 가능


#딥러닝모델(히든레이어들 있고.. 계산해서 마지막 결과물 보여주는것..) 쉽게 만들어줌 
"""
tf.keras.models.Sequential({
    레이어1,
    레이어2,
    레이어3
})
"""
# Sequential함수쓰면 신경망 레이어들 쉽게 만들어줌





"""
1. 딥러닝 모델만들기
"""
model = tf.keras.models.Sequential({
    tf.keras.layers.Dense(64, activation='tanh'),   #첫번째 레이어임. 레이어만드는 공식임. 파라미터값은 그 레이어에 들어가는 노드갯수임. 
                                 # 노드갯수는 결과 잘 나올때까지 실험적으로 때려박아 파악해야하는거임. 근데 관습적으로 2의제곱수를 많이 넣음 
    tf.keras.layers.Dense(128, activation='tanh'), 
    tf.keras.layers.Dense(1, activation='sigmoid'),   # 마지막 결과는 0또는 1 등 하나일경우엔 1 적으면됨. 근데 결과를 여러개로 다양하게 만들고싶다면 1이상 놓으면됨 
})
# 정확히는 0과 1은 아니고 대학입학 확률예측이므로 0과1사이의 수가 나올거임

# 두번째 파라미터값엔 활성함수 넣어주면됨. 활성함수엔 sigmoid, tanh, relu, softmax 등이 있음.
# 근데 마지막 레이어는 항상 예측결과를 뱉어내야함. 그래서 활성함수를 sigmoid넣으면됨. 이건 모든 값을 0과1사이로 압축해서 뱉어줌. 



"""
2. 모델 컴파일하기
"""
model.compile(optimizer = 'adam' , loss ='binary_crossentropy' , metrics=['accuracy'])
# 옵티마이저란? - 경사하강법으로 기울기를 빼는 등의 방법으로 w값을 계속 수정한다고 했잖음. 그때 빼는 그 기울기값을 알아서 잘 조정해주는 녀석임. 항상 균등한 learning rate로 빼면 w값 조정이 잘안됨. 그래서
# 상황에 맞게 조정해서 빼주는 녀석임. 옵티마이저는 adam, adagrad 등 여러가지 있는데 걍 기본적으로 adam 쓰면됨

# 손실함수는 mean squared error등 여러가지 있음. 근데 결과가 0과 1사이의 분류/확률예측 문제에선 binary_crossentropy 이거씀.
# 그냥 내가 원하는 결과값이 0과 1 사이라면 이거 쓴다고 알면됨 

# metrics 이건 그냥 딥러닝 모델 평가할때 어떤 요소로 평가할건지 정해주는 녀석이라고 생각하면됨. 그냥 저렇게 accuracy적으면됨.




"""
3. 모델 학습(fit)시키기
"""
model.fit(x데이터, y데이터, epochs=10) # epochs는 미리 준비한 데이터를 빠꾸시켜가면서 몇번 학습시킬지 정함

# x데이터에는 인풋값(성적, 등수 등) , y데이터에는 결과값 (합격인지 아닌지.. 0과 1로...) 넣어주면됨.
