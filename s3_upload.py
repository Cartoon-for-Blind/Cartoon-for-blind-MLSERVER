import boto3
import cv2
import numpy as np
import requests
import os
from dotenv import load_dotenv

# .env 파일에서 AWS 키 가져옴
load_dotenv()
key_id = os.environ.get("aws_access_key_id")
secret_key = os.environ.get("aws_secret_access_key")

def s3_connection():
    try:
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key,
        )
    except Exception as e:
        print(e)
    else:
        return s3

s3 = s3_connection()

def imread_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Failed to download image from URL: {url}")
        return None