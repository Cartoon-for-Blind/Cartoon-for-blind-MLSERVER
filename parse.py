import json
import re

def add_index(panels): # penel_index필드 추가해서 0부터 채우기
    if isinstance(panels, str):
        panels = json.loads(panels) 

    for i, panel in enumerate(panels):
        panel["panel_index"] = i  
    
    return json.dumps(panels, indent=4, ensure_ascii=False)

def add_threadid(panels, thread_id="0"):  
    # 같은 책 내용을 이어가기 위해 같은 thread_id를 저장해놓고 사용해야함
    # thread_id필드 추가해서 채우기, 백엔드 db에 저장해놓고 사용
    if isinstance(panels, str):
        panels = json.loads(panels) 

    for i, panel in enumerate(panels):
        panel["thread_id"] = thread_id
    
    return json.dumps(panels, indent=4, ensure_ascii=False)

def parse_texts(texts): # JSON 파싱용
    json_objects = [json.loads(item) for sublist in texts for item in sublist]
    return json_objects