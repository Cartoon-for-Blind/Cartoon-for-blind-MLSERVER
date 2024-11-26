import os
import cv2
import time
import json
import re
import subprocess
import pyautogui
import matplotlib.pyplot as plt
from ultralytics import YOLO
from yolov8_bubbles import * # bubble_detect(), bubble_on_panel(), text_on_bubble(), text_on_bubble_on_panel()
from s3_upload import *      # imread_url()
from clova_ocr import *      # image_ocr()


def split_image(name): # 이미지 자르기
    # image = imread_url(f"https://meowyeokbucket.s3.ap-northeast-2.amazonaws.com/comics/Pages/{name}.jpg") # 확장자 같이써줘야함
    image_path = f"C:\\Users\\vkdnj\\Zolph\\comics\\proc_images\\{name}.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    height, width, _ = image.shape
    
    # 가로가 세로보다 길면 2페이지라 생각하고 자르고 아니면 1페이지라고 인식 (left, keep으로 구분)
    if width > height:
        mid_width = width // 2
        left_image = image[:, :mid_width]   
        right_image = image[:, mid_width:]  
        
        # 자른거 _left, _right로 나눠 따로 저장
        left_image_path = f'C:\\Users\\vkdnj\\Zolph\\comics\\proc_images\\{name}_left.jpg'
        right_image_path = f'C:\\Users\\vkdnj\\Zolph\\comics\\proc_images\\{name}_right.jpg'
        cv2.imwrite(left_image_path, left_image)
        cv2.imwrite(right_image_path, right_image)
        # S3에도 업로드
        s3.upload_file(left_image_path, "meowyeokbucket", f"comics/proc_images/{name}_left.jpg")
        s3.upload_file(right_image_path, "meowyeokbucket", f"comics/proc_images/{name}_right.jpg")
        return "left"
    
    else:
        keep_image_path = f'C:\\Users\\vkdnj\\Zolph\\comics\\proc_images\\{name}_keep.jpg'
        cv2.imwrite(keep_image_path, image)
        s3.upload_file(keep_image_path, "meowyeokbucket", f"comics/proc_images/{name}_keep.jpg")
        return "keep"

def open_directory(path):
    if os.path.exists(path):
        subprocess.run(f'explorer "{path}"', shell=True)
        time.sleep(1) 
        pyautogui.press('f11')
        pyautogui.hotkey('ctrl', 'shift', '1')
        time.sleep(1) 
    else:
        print(f"Error: The directory '{path}' does not exist.")

def sort_panels(boxes_with_objects, y_threshold=20, y_weight=1.5):

    # x1 + y_weight * y1 기준으로 초기 정렬
    boxes_with_objects.sort(key=lambda item: item[0][0] + y_weight * item[0][1])

    # 비슷한 x1 + y_weight * y1 값 그룹으로 나누기
    grouped_boxes = []
    current_group = [boxes_with_objects[0]]
    
    for i in range(1, len(boxes_with_objects)):
        current_box = boxes_with_objects[i][0]
        last_box = boxes_with_objects[i - 1][0]

        # 현재 박스와 이전 박스의 y1 값 차이 확인
        y1_current, y1_last = current_box[1], last_box[1]
        
        # y 값이 임계값 이내라면 같은 그룹에 추가
        if abs(y1_current - y1_last) <= y_threshold:
            current_group.append(boxes_with_objects[i])
        else:
            # 새로운 그룹 생성
            grouped_boxes.append(current_group)
            current_group = [boxes_with_objects[i]]
    
    # 마지막 그룹 추가
    if current_group:
        grouped_boxes.append(current_group)

    # 그룹 내 정렬 (x1 + y_weight * y1 기준)
    for group in grouped_boxes:
        group.sort(key=lambda item: item[0][0] + y_weight * item[0][1])

    # 그룹을 flatten하여 반환
    sorted_boxes = [item for group in grouped_boxes for item in group]
    return sorted_boxes


def calculate_iou(box1, box2):
    # 박스 좌표 추출
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 교차 영역 계산
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    # 교차 영역의 너비와 높이 계산
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # 각 박스의 면적 계산
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # IoU 계산
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def remove_duplicate_panels(boxes_with_objects, iou_threshold=0.5):
    filtered_boxes = []
    for i, (box1, obj1) in enumerate(boxes_with_objects):
        is_duplicate = False
        for j, (box2, obj2) in enumerate(filtered_boxes):
            # IoU 계산 및 중복 여부 확인
            iou = calculate_iou(box1, box2)
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_boxes.append((box1, obj1))
    return filtered_boxes

def panel_seg(name, y_threshold=20, iou_threshold=0.5): # 컷 자르기
    model = YOLO('C:\\Users\\vkdnj\\Zolph\\models\\panel_best.pt') 
    
    image = imread_url(f"https://meowyeokbucket.s3.ap-northeast-2.amazonaws.com/comics/proc_images/{name}.jpg")
    results = model(image, conf=0.4)
    
    # 탐지된 패널 좌표 및 이미지를 저장
    boxes_with_objects = []
    for idx, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        detected_object = results[0].orig_img[y1:y2, x1:x2]
        boxes_with_objects.append(((x1, y1, x2, y2), detected_object))

    # 중복 패널 제거
    boxes_with_objects = remove_duplicate_panels(boxes_with_objects, iou_threshold)
    
    # 박스 정렬
    sorted_boxes = sort_panels(boxes_with_objects, y_threshold)
    panel_coords_list = [(box[0][0], box[0][1], box[0][2], box[0][3]) for box in sorted_boxes]

    # 결과 저장 경로 및 업로드 처리 (이전 코드와 동일)
    folder_path = f"C:\\Users\\vkdnj\\Zolph\\comics\\panel_seg\\{name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    open_directory(folder_path)

    for idx, (box, detected_object) in enumerate(sorted_boxes):
        file_path = os.path.join(folder_path, f"{name}_{idx}.jpg")
        cv2.imwrite(file_path, detected_object)
        s3.upload_file(file_path,"meowyeokbucket",f"comics/panel_seg/{name}/{name}_{idx}.jpg")

    result_plotted = results[0].plot()
    cv2.imwrite(f"{folder_path}\\{name}.jpg", result_plotted)
    s3.upload_file(f"{folder_path}\\{name}.jpg","meowyeokbucket",f"comics/panel_seg/{name}/{name}.jpg")
    
    return panel_coords_list


def get_text(name): # 위의 함수들을 모두 이용해서 텍스트 추출
    panel_coords_list = panel_seg(name)
    bubble_coords_list = bubble_detect(name)
    text_coords_list = image_ocr(name)
    
    classified_bubbles = bubble_on_panel(bubble_coords_list, panel_coords_list)
    classified_texts = text_on_bubble(bubble_coords_list, text_coords_list)
    panel_texts = text_on_bubble_on_panel(classified_bubbles, classified_texts)
    
    texts = []
    for outer_key in sorted(panel_texts.keys()):
        inner_list = []
        for inner_key in sorted(panel_texts[outer_key].keys()):
            if panel_texts[outer_key][inner_key]:  # 비어 있지 않은 경우에만 추가
                inner_list.append(" ".join(panel_texts[outer_key][inner_key]))
        texts.append(inner_list)

    return texts