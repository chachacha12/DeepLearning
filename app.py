"""
개/고양이 구분하는 ai 만들어볼거임
kaggle 사이트에서 compete 에 들어가서 dog vs cat 검색해서 개 고양이 관련 이미지 3만장 정도 가져올거임. 들어가서 data 들어가서 download all 해줌.
(근데 그냥 코랩에서 코드 명령어로 가져올거면 지금 다운로드 안해줘도 될듯?)

@만약 pc가 아니라서 ram이나 메모리 등 딸릴거같으면 구글 colab이용해서 구글의 컴퓨팅파워 이용해도됨.
구글드라이브 - 파일하나생성(난 파이썬으로 딥러닝 작업 파일 만듬)해 - 마우스 우클릭해서 더보기... 구글colab ㄱㄱ 

구글colab에선 kaggle등에서 다운받은 이미지 업로드 어떻게 하나?
1. 파일트리 열어서 이미지 거기에 업로드 하든가.. -> 시간 너무 오래걸림
2. 직접 다운받는 코드를 짜든가..



2번으로 다운 할거임. 

 - 일단 kaggle사이트에서 오른쪽상단의 내 프로필누르고 마이프로필 - account 로 들어감.
 - API 부분에 create new token 클릭 - kaggle.json파일 다운받아질거임.  
 - 그 파일을 코랩에 드래그해서 왼족의 파일트리 빈공간에 끌어다두면 됨.

새로운 zip파일이 생길거임. 
코랩에서 압축푸는법:  !unzip -q train.zip -d .      (이렇게 코랩에 코드적고 이 코드만 실행하면 train.zip이라는 파일이 압축풀림.)

쨋든 이런식으로해서 젤 첨에 있던 dogs-vs-cats-redux-kernels-edition.zip이 파일 압축풀고 그 후에 train.zip폴더 압축풀어주기. 
(test.zip 압축 안푸는 이유는 테스트용 데이터들이라 label값이 따로 안붙어있음..개인지 고양이인지 안붙어있음. 그래서 이걸 validation이라던지 테스트용으로 쓸 순 없음. 쨋든 중요x라서 스킵)

(여기까지 이제 데이터 kaggle에서 가져와서 업로드 하는 과정 끝.   만약 코랩안쓰고 로컬환경에서 개발하는 거면 알아서 압축푸셈)

"""



"""
이때까지 코랩에 작성해준 코드---------------------------------------

 import os

os.environ['KAGGLE_CONFIG_DIR'] = '/content/'  # 다운받은 kaggle.json파일의 경로값을 넣어줌.

!kaggle competitions download -c dogs-vs-cats-redux-kernels-edition

#여기까지 진행하고 실행해봤는데 에러뜨면 kaggle에서 내 account에서 create new api버튼 다시 눌러서 새로운 kaggle.json파일 붙여넣기


!unzip -q dogs-vs-cats-redux-kernels-edition.zip -d .

!unzip -q train.zip -d .


print( len( os.listdir('/content/train/'))) # 이렇게 해주면 해당 경로안의 이미지파일이 몇개인지 출력 
# 참고로 os.listdir('/content/train/') 여기까지만 쓰면 이 폴더안의 파일을 리스트안에 다 담아줌 -> [파일명1, 파일명2, 파일명3....]
# 그래서 len()함수는 그 리스트안의 데이터 갯수를 세었을뿐임

-----------------------------------------------------------------------------


"""
