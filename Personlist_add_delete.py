#새로운 얼굴을 dataframe에 추가하는 function. dataframe 다른 형태 쓸 경우, 변수는 건들이지 말고 dataframe 부분만 건들이면됨.
# dataframe 의견 : 어차피 전체적인 list 구성해야하니까, 이 형태로 (아래 만들어놓은거) 가는게 좋지 않을까?
import pandas as pd
import os
from pandas import DataFrame as df
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
# 프로그램을 통해 이름이랑 저장되어있는 경로를 입력받음
# 추가버튼을 눌렀다 --> API 서버와 통신해서 face_id를 받는다. (함수호출)
# 이름은 사용자에게 직접 입력받기.
# 파일을 선택해서 파일 경로가 여기로 들어가게 해야함. text로 말고.
def add_person_to_list(name,directory_path):

    df1 = pd.read_csv('test_file.csv')
    #이건 기존 파일에 뭐있는지 확인하기 위한 용도
    print(df1)
    #API에 접근하기 위해 객체를 생성함. 이 객체는 프로그램 켜지면 한 번 생성하고 안없어지게 해야함! 이건 함수 작동하는거 보여주려고 넣어놓음.

    KEY = os.environ['FACE_SUBSCRIPTION_KEY']
    ENDPOINT = os.environ['FACE_ENDPOINT']
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    #주의! 사진은 반드시 한명만 나와있는거로만 해야함 이거에선.
    image_path = directory_path
    image_open = open(image_path, 'rb')
    face_client.face_list.add_face_from_stream(face_list_id='test_list',image=image_open)

    face_list = face_client.face_list.get(face_list_id='test_list')
    size = (len(face_list.persisted_faces))-1
    get_face_id = face_list.persisted_faces[size].persisted_face_id
    #데이터 추가 하는 곳 (간단하게 loc을 이용해 추가함)
    data= {'이름': name, 'face_id': get_face_id, '파일경로': directory_path}
    df2 = df1.append(data,ignore_index=True)
    print(df2)
    df2.to_csv("C:/Users/CNH/PycharmProjects/untitled1/test_file.csv", index= False)

#print("이름을 입력해주세요")
#name = input()
#print("저장되어 있는 경로는요?")
#directory_path= input()
#add_person_to_list(name= name, directory_path=directory_path)

# 이 함수 활용하기 위해서는 어떤 사람을 지울껀지 선택하면 index값 return 해주는 함수 필요함.
def delete_person_from_list(index):
    KEY = os.environ['FACE_SUBSCRIPTION_KEY']
    ENDPOINT = os.environ['FACE_ENDPOINT']
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
    df1 = pd.read_csv('test_file.csv')
    del_face_id = df1.loc[index,"face_id"]
    print(del_face_id)
    face_client.face_list.delete_face(face_list_id='test_list',
                                      persisted_face_id=del_face_id)
    A = face_client.face_list.get(face_list_id='test_list')
    for i in range (len(A.persisted_faces)) :
        print(A.persisted_faces[i].persisted_face_id)
    df2 = df1.drop(index)
    print(df2)
    df2.to_csv("C:/Users/CNH/PycharmProjects/untitled1/test_file.csv", index=False)



