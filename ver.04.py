import threading
import datetime
import os
import signal
import cv2
import numpy as np
import pandas as pd
import asyncio
import io
import glob
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, \
    OperationStatusType

def logging(persisted_face_id):
        #df1은 설정값을 읽어와야함. 안찾는 친구는 굳이 반복문 돌릴 필요 없으니까
    df1 = pd.read_csv('./data/person_list.csv')
    df2 = df1.loc[df1['face_id'] == persisted_face_id]

    if(df2.empty):
     print("그런사람 없어요.")

    elif(df2.empty == False ):
     log_file = pd.read_csv('./data/log_file.csv')
     temp = df2.copy(deep = True)
     time = datetime.datetime.now()
     now = time.strftime('%Y-%m-%d %H:%M:%S')
     temp["Time"] = now
     log_file = log_file.append(temp, ignore_index=True)
     log_file.to_csv("C:/Users/CNH/PycharmProjects/untitled1/log_file.csv", index= False)
     print("기록")

#함수정의 시작
def API_CALL():
    path = "./outputs/"
    multi_face_image_path = path + 'mnist_merged' + str(int(count / (set_num - 1))) + ".jpg"
    multi_image_name = os.path.basename(multi_face_image_path)
    image_name_2 = open(multi_face_image_path, 'rb')
    detected_faces2 = face_client.face.detect_with_stream(image=image_name_2, return_face_id=True,
                                                          recognition_model='recognition_03')

    if detected_faces2:
        print('Detected face ID from', multi_image_name, ':')
        for face in detected_faces2:
            print("발견"+face.face_id)
            similar_faces = face_client.face.find_similar(face_id=face.face_id,
                                                          face_list_id='api_list')
            # 의미없는 반복문이긴 하나, SIMILAR_FACE가 없는 경우를 위해서 만듬. 
            for i in similar_faces:
                #log 기록하는 csv 파일 열기 , list 포함하고있는 파일도 열기
                if (similar_faces[0].confidence > 0.6):
                    logging(similar_faces[0].persisted_face_id)


            img = cv2.imread(multi_face_image_path)
            img = img[x - 10:x + w + 10, y - 10:y + h + 10]
            # cv2.imshow("linear", img)
#함수정의 끝
#프로그램 시작

# 몇장을 한번에 합쳐서 검출한 것인지는, set_num 변수값을 지정해서 바꿔주면됩니다!
set_num = 4
# 1. azure api에 접근하기 위한 object들을 key값을 이용해 생성해줌
KEY = os.environ['FACE_SUBSCRIPTION_KEY']
ENDPOINT = os.environ['FACE_ENDPOINT']

# 2. Load Yolo( yolo 알고리즘 동작에 필요한 weight 파일들을 불러옴)
net = cv2.dnn.readNet("./data/weights/yolov3-wider_16000.weights", "./data/cfg/yolov3-face.cfg")
classes = []
with open("./data/names/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 이미지 이름을 변경해주기 위한 count 변수
count_1 = 0
count = 0
# 3. 특정인을 찾기위해 특정인 사진을 입력받음
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# Display the detected face ID in the first single-face image.
# 3-1 IDs are used for comparison to faces (their IDs) detected in other images.

# 해당 이미지에서 검출된 얼굴들이 모두 저장됨. (face라는 객체에 face_id라는 string 변수가 잇는듯. detected_face = list )
# 3-2 Save this ID to use in Find Similar -> 이 코드는 id 한개만 저장

# 4. load video
cap = cv2.VideoCapture('./123.mp4')
cv2.dnn.DNN_TARGET_OPENCL

#flag의 용도: flag는 처음 영상 시작시 합쳐지는 파일이 만들어지는 규칙의 예외이기 때문에 설정해줌.
# 이건 반복문. 영상이 살아있거나, 특정 키를 누르기 전까지 계속 찾는걸 반복함.
flag = False
#시간을 저장하는 변수.
time = datetime.datetime.now()
now_time = time.strftime('%Y-%m-%d %H:%M:%S')


while True:
    # capture video
    print("한프레임")
    ret, frame = cap.read()
    # 현재 프레임 == 총 프레임 : 비디오의 끝 종료
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        print("video end")
        break

    # 프레임 수 조절 (Test 중에는 안씀)
    # if cap.get(cv2.CAP_PROP_POS_FRAMES) % 15 != 0:
    # continue
    frame = cv2.resize(frame, None, fx=1, fy=1)
    height, width, channels = frame.shape
    # 5. Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    path = "./outputs/"

    # print(indexes) 이런거 저장됏다고 보여주는 구문
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # 박스 태그 (지우고 돌릴 것)
            # label = str(count)
            # 파일 이름
            name = str(count_1) + ".jpg"
            # print(label, "(", x, y, ")", ",", "(", x + w, y + h, ")")
            color = colors[i]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(frame, label, (x, y + 30), font, 1, color, 1)

            # 얼굴 자른 후 outputs 폴더에 저장
            faceimg = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            if not faceimg.any():
                continue
            faceimg = cv2.resize(faceimg, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
            # cv2.imshow("linear", faceimg)
            # cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(path + name, faceimg)
            # cv2.imshow("Image", frame)
            print(count_1)
            im0 = Image.open(path + name)

            if count % set_num == 0:
                if flag == False:
                    merged = Image.new('L', (200 * set_num, 200 * 1))
                    flag = True
                    print("생성")

                elif flag == True:
                    print("저장")
                    merged.save(path + 'mnist_merged' + str(int(count/(set_num-1))) + ".jpg")
                    merged = Image.new('L', (200 * set_num, 200 * 1))
                    print("생성, flag = true")
                    API_CALL()

                merged.paste(im0, (200 * (count % set_num), 0))
                count += 1
                count_1 += 1

            elif count % set_num != 0:
                temp_time = datetime.datetime.now()
                temp_now_time = temp_time.strftime('%Y-%m-%d %H:%M:%S')
                if(now_time != temp_now_time):
                    merged.paste(im0, (200 * (count % set_num), 0))
                    now_time = temp_now_time
                    count+= set_num-(count%set_num)
                    count_1+= 1
                    print("저장")
                    merged.save(path + 'mnist_merged' + str(int(count/(set_num-1))) + ".jpg")
                    API_CALL()
                    flag = False
                elif(now_time == temp_now_time):
                    count_1 += 1
                    count += 1
                    merged.paste(im0, (200 * (count % set_num), 0))


            #시간 계속 계산해서 1초 넘어가면 자르고 그냥 API 넘겨버리게 하기.


    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break
# 프레임이 끝나게 되면 끝난 부분에서도 다시 저장해 similar 한 face가 있는지 검사해야하므로, 똑같은 부분이 나오게됨.
path = "./outputs/"
merged.save(path + 'mnist_merged' + str(count / (set_num)) + ".jpg")
cap.release()
print("program end")
os.kill(os.getpid(), signal.SIGTERM)
