import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from s3_upload import * #imread_url()


def bubble_detect(name, margin=10): # 말풍선 감지
    model = YOLO('C:\\Users\\vkdnj\\Zolph\\models\\comic-speech-bubble-detector.pt')  
    image = imread_url(f"https://meowyeokbucket.s3.ap-northeast-2.amazonaws.com/comics/proc_images/{name}.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, conf=0.4)
    
    # 리스트에 감지된 말풍선의 좌표 저장
    bubble_coords_list = []

    for result in results: 
        boxes = result.boxes.xyxy  # x_min, y_min, x_max, y_max 좌표 값
        
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box[:4]
            
            x_min = round(x_min.item())
            y_min = round(y_min.item())
            x_max = round(x_max.item())
            y_max = round(y_max.item())
            
            # 말풍선 좌표 리스트에 추가
            bubble_coords_list.append((x_min, y_min, x_max, y_max))

    # 중심 좌표와 오차 범위를 고려한 정렬
    # y 좌표 우선 정렬 후, y 좌표가 같을 경우 x 좌표로 정렬
    bubble_coords_list.sort(key=lambda coords: (coords[1], (coords[0] + coords[2]) / 2))  # y_min 우선 정렬

    # 말풍선 표시된 이미지
    annotated_image = results[0].plot()
    
    # 원본 이미지에 탐지 결과를 덮어쓰고 저장
    cv2.imwrite(f"C:\\Users\\vkdnj\\Zolph\\comics\\panel_seg\\{name}\\{name}_bubbles.jpg", annotated_image)
    # s3에도 업로드
    s3.upload_file(f"C:\\Users\\vkdnj\\Zolph\\comics\\panel_seg\\{name}\\{name}_bubbles.jpg","meowyeokbucket",f"comics/panel_seg/{name}/{name}_bubbles.jpg")

    return bubble_coords_list


def bubble_on_panel(bubble_coords_list, cut_coords_list): # 컷 별로 말풍선 그룹화
    
    classified_bubbles = {i: [] for i in range(len(cut_coords_list))} 

    # 각 말풍선을 컷에 분류
    for bubble_idx, bubble_coords in enumerate(bubble_coords_list):
        bubble_x_min, bubble_y_min, bubble_x_max, bubble_y_max = bubble_coords
        
        # 말풍선 중심 좌표 계산
        bubble_x_center = (bubble_x_min + bubble_x_max) / 2
        bubble_y_center = (bubble_y_min + bubble_y_max) / 2
        
        # 각 컷의 좌표와 비교
        for cut_idx, cut_coords in enumerate(cut_coords_list):
            cut_x_min, cut_y_min, cut_x_max, cut_y_max = cut_coords
            
            # 말풍선의 중심이 컷 안에 있으면 해당 컷에 분류
            if (cut_x_min <= bubble_x_center <= cut_x_max and
                cut_y_min <= bubble_y_center <= cut_y_max):
                classified_bubbles[cut_idx].append(bubble_idx) 
        
    return classified_bubbles


def text_on_bubble(bubble_coords_list, texts_with_positions): # 말풍선별로 텍스트 그룹화
    classified_texts = {i: [] for i in range(len(bubble_coords_list))}  

    # 각 텍스트를 말풍선에 분류
    for text_info in texts_with_positions:
        text_x, text_y = text_info[0], text_info[1]  # 튜플의 첫 번째와 두 번째 요소가 x, y 좌표
        text_content = text_info[2]  # 세 번째 요소가 텍스트 내용
        
        for bubble_idx, bubble_coords in enumerate(bubble_coords_list):
            bubble_x_min, bubble_y_min, bubble_x_max, bubble_y_max = bubble_coords
            
            # 텍스트 좌표가 말풍선 영역 내에 있는지 확인
            if (bubble_x_min <= text_x <= bubble_x_max and
                bubble_y_min <= text_y <= bubble_y_max):
                classified_texts[bubble_idx].append(text_content) 

    return classified_texts


def text_on_bubble_on_panel(classified_bubbles, classified_texts): # 위의 함수들을 모두 이용해 텍스트 순서 맞춰서 2차원 리스트형태로 리턴
    panel_texts = {}
    bubble_global_index = 0 

    for panel_idx, bubbles in classified_bubbles.items():
        panel_texts[panel_idx] = {}
        for bubble_coords in bubbles:
            # classified_texts의 각 말풍선에 속하는 텍스트들을 패널에 따라 분류
            if bubble_global_index in classified_texts:
                panel_texts[panel_idx][bubble_global_index] = classified_texts[bubble_global_index]
            else:
                panel_texts[panel_idx][bubble_global_index] = []  # 텍스트가 없으면 빈 리스트로

            bubble_global_index += 1  # 다음 말풍선을 위한 인덱스 증가

    return panel_texts