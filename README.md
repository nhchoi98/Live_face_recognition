# Live_face_recognition
Live_face_recognition
1. 프로그램의 목적: 사진을 입력받아 기존에 입력되있는 사진속 인물과 동일인인지를 판별한다. 

2. 프로그램의 앞으로 개발 방향: CCTV의 영상을 지속적으로 입력받아, 사진 속 인물과 동일인이 나타나는지를 판단해주는 프로그램을 제작해야함 

3. azure api 사용을 위해선 Azure SDK LIBRARY가 설치 되어있어야함 (pip install azure-storage-blob, pip install --upgrade azure-cognitiveservices-vision-face)


<07/20>
1. face라는 객체에, face_id라는 string 변수 member가 존재하며, 한 사진을 분석시 프로그램이 detected_face는 list의 형태로 저장하는듯함. 
--> 다수의 리스트 멤버가 생길 수 있으므로, 한 변수에 대해 다수의 리스트 멤버를 비교하는게 필요함. 

2. 제공 받는 파일이 다수의 얼굴 (멤버)가 포함되었을 수 있으므로, 이 중에 선택하는 옵션을 넣는게 좋다 사료됨. 

3. 다른 문제점: 픽셀이 커서 그런지, 얼굴 탐색하는데 시간이 좀 걸리는듯 함. 욜로 알고리즘과 합쳐서 시간 비교를 하는게 필요해보임 
