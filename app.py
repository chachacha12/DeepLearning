
"""
하단 코드는 모두 구글코랩에서 작성한 것임. 로컬환경에서 돌리기엔 ram 부족할까봐.
"""

import os
import tensorflow as tf
import shutil # 파이썬 shutil 라이브러리임. 여기선 이미지파일을 다른 폴더로 이동시키기 위해 사용 

print( len( os.listdir('/content/train/'))) 

"""
(이미지 2만장 3초만에 Dataset 만들기)

* 근데 구글 코랩의 content경로(바로 처음에 나오는 경로)안에 파일 만드는 것들은 이 코랩끄면 사라지는듯.


저번 강의에선 kaggle에서 jpg 이미지들 2만장 파일을 가져오는것까지 했었음. 이제 가져온 그 이미지들을 숫자이미지로 변경해야함.

1. 파이썬의 opencv라이브러리로 이미지 숫자화 하기 (반복문이 많이 필요할듯) 
   또는 tf.keras 이용해서 한줄만에 처리하기

    --> 편하게 tf.keras 이용해서 해볼거임.
"""

"""
일단 개, 고양이사진을 각각 다른 폴더로 분류해서 저장해주기. (현재 경로에 dataset이란 폴더 새로 만들고, 그 안에 cat, dog폴더 또 만들어줌 일단)

1. 일단 train안의 모든 파일명 출력해보고
2. 파일명에 cat 들어있으면 cat폴더로, dog들어있으면 dog폴더로 이동

@@@ 이미지 classification 딥러닝할 땐, 이렇게 dataset이란 폴더안에 카테고리별로 사진을 분류해서 시작하는게 국룰임. 이래야 데이터뽑기 쉽기 때문임.
"""

# 이렇게 하면 i는 cat1.jpg.  cat2.jpg   dog1.jpg... 등의 이미지들이 하나씩 들어갈거임
for i in os.listdir('/content/train/'):
  if 'cat' in i:
    shutil.copyfile('/content/train/' + i, '/content/dataset/cat/'+i)       # shutil.copyfile (어떤파일을, 어떤 경로로) # 파일을 다른 경로로 복사해서 붙여주는 명령 
  if 'dog' in i:
    shutil.copyfile('/content/train/' + i, '/content/dataset/dog/'+i)                                                                          # 코랩말고 로컬컴퓨터에서 하는경우는 경로의 맨앞 슬래쉬 빼고, 로컬경로 잘 잡아주면됨.
                                                                            

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

"""
validation 데이터 만들어서 테스트 하는 방법

1. 위처럼 데이터셋을 2개 연달아서 만들어줌. (그냥 복붙해주면됨)
2. subset에 이름 지어줌. 위에가 트레이닝 데이터고 아래가 validation 데이터임.
3. validation_split에다가 몇프로의 데이터를 쪼개서 validation에 이용할건지 적어줌 
(0.2라고 적으면 위의 train_ds는 80%의 데이터셋을 가진거고 아래의 val_ds는 20%의 데이터셋을 가진거임)
(보통 20프로정도를 validation데이터셋으로 가지는게 관례같은거임)

4. seed값 아무거나 두 곳 다 똑같은 값 적어줌 (그냥 둘 다 동일한 갯수의 batch를 뽑아 쓸거라는 의미인듯)

"""

print(train_ds)   # ( (숫자이미지2만개), (0이나 1로된 정답2만개) )   이런식의 shape일거임