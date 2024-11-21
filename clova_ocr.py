import requests
import json
import time
import base64
import os
from dotenv import load_dotenv

load_dotenv()
clova_key = os.environ.get("clova_key")
clova_url = os.environ.get("clova_url")

def download_image(name): # S3에서 이미지 다운로드
    image_url = f"https://meowyeokbucket.s3.ap-northeast-2.amazonaws.com/comics/proc_images/{name}.jpg"
    image_path = f'C:\\Users\\vkdnj\\Zolph\\comics\\proc_images\\{name}.jpg'

    response = requests.get(image_url)
    
    if response.status_code == 200:
        with open(image_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
    return image_path

def image_ocr(name): # 클로바 ocr 때리는 함수
    image_path = download_image(name)
    url = clova_url
    headers = {
        "Content-Type": "application/json",
        "X-OCR-SECRET": clova_key
    }
    
    # 현재 timestamp 생성
    timestamp = int(time.time() * 1000)
    
    # 로컬 이미지 파일을 읽고 Base64로 인코딩
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    # API에 보낼 데이터 구성
    data = {
        "images": [
            {
                "format": "png",  # 이미지 포맷에 맞춰서 변경
                "name": "medium",
                "data": image_base64  # Base64 인코딩된 이미지 데이터
            }
        ],
        "lang": "ko",
        "requestId": "string",
        "resultType": "string",
        "timestamp": timestamp,
        "version": "V1"
    }
    
    # POST 요청 보내기
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # 응답 결과 처리
    if response.status_code == 200:
        response_data = response.json()
    
        # 'fields'에서 'inferText'와 'boundingPoly' 추출
        fields = response_data["images"][0]["fields"]
        
        # inferText와 boundingPoly의 가장 왼쪽 위 좌표 (x, y) 추출 및 텍스트 포함
        texts_with_positions = [
            (
                field["boundingPoly"]["vertices"][0]["x"],  # x 좌표
                field["boundingPoly"]["vertices"][0]["y"],  # y 좌표
                field["inferText"]  # 텍스트 내용
            )
            for field in fields
        ]
    
        # 오차 범위 설정
        y_tolerance = 20  # y 좌표 오차 범위
    
        # y 값을 오차 범위 내에서 그룹화하기 위한 정렬
        def sort_key(item):
            return (round(item[1] / y_tolerance), item[0])
    
        # 정렬: y 범위 내에서 x 기준으로 정렬
        texts_with_positions.sort(key=sort_key)
    
        # 정렬된 좌표 리스트 반환
        return texts_with_positions
    
    else:
        print("Error:", response.status_code)
        print("Message:", response.text)
        return []