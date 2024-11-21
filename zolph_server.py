from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from yolov8_bubbles import * # bubble_detect(), bubble_on_panel(), text_on_bubble(), text_on_bubble_on_panel()
from yolov8_panel import *   # split_image(), panel_seg(), get_text()
from clova_ocr import *      # image_ocr()
from assistants_api import * # new_assistant(), new_book(), assistant_image_captioning()
from s3_upload import *      # imread_url()
from parse import *          # add_index()
from document_extraction import * # process_image()


app = Flask(__name__)

UPLOAD_FOLDER = "C:\\Users\\vkdnj\\Zolph\\comics\\Pages\\"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_no_dialogue(texts, messages):
    for i, message_group in enumerate(messages):
        # texts[i]가 비어있지 않은 경우 건너뜀
        if texts[i]:
            continue
        # texts[i]가 비어있는 경우, 첫 번째 메시지에 대해 수정 수행
        if message_group:
            data = json.loads(message_group[0])  # 첫 번째 메시지를 JSON으로 변환
            dialogue_entries = data.get("dialogue", [])
            # 모든 dialogue 값을 "[No dialogue]"로 설정
            for entry in dialogue_entries:
                for key in entry:
                    entry[key] = "[No dialogue]"
            # 수정된 데이터를 다시 JSON 문자열로 변환하여 덮어쓰기
            messages[i][0] = json.dumps(data, ensure_ascii=False)
            # 나머지 메시지는 삭제
            messages[i] = messages[i][:1]
    return messages

def get_result(image_name, is_new_assidtant = False, id = "0") :
    
    # 새책이면 새로운 thread_id 생성, 아니면 기존거 가져옴
    if id=="0":
        thread_id = new_book()
    else : thread_id = id
    
    if is_new_assidtant == True :
        assistant_id = new_assistant()
        print(assistant_id)
    else : 
        assistant_id = "asst_dxz93PwBYsW3CasEKzuQZUID"
        
    cropped_image_name = process_image(image_name) # cropped_image로 처리후 "cropped_imagename" 리턴
    split_res = split_image(cropped_image_name) # cropped image -> left,right or keep으로 처리후 "left" or "keep" 등 리턴
    
    if split_res == "keep" : 
        proc_image_name = cropped_image_name+"_keep" 
        texts = get_text(proc_image_name)
        messages = assistant_image_captioning(proc_image_name, texts, assistant_id, thread_id)
        print(texts)
        print(messages)
        messages = check_no_dialogue(texts, messages)
        print(messages)
        parsed_texts = parse_texts(messages) # json내용 정리
        index_added = add_index(parsed_texts) # panel_index필드 추가
        res = add_threadid(index_added, thread_id) # thread_id필드 추가
        return res, thread_id
    
    else :
        left_name, right_name = cropped_image_name+"_left", cropped_image_name+"_right"
        texts_left = get_text( left_name )
        texts_right = get_text( right_name )
        
        messages_left = assistant_image_captioning(left_name, texts_left, assistant_id, thread_id)
        messages_right = assistant_image_captioning(right_name, texts_right, assistant_id, thread_id)
        
        messages = messages_left + messages_right
        print(messages)
        messages = check_no_dialogue(texts_left+texts_right, messages)
        print(messages)
        parsed_texts = parse_texts(messages) # json내용 정리
        index_added = add_index(parsed_texts) # panel_index필드 추가
        res = add_threadid(index_added, thread_id) # thread_id필드 추가
        return res, thread_id

# 이미지 대본 생성용
@app.route('/upload', methods=['POST'])
def upload_image():
    # 파일이 있는지 체크
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    thread_id = request.form.get('thread_id')
    if not thread_id:
        return jsonify({'error': 'No thread_id provided'}), 400
    
    # 파일 저장
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image_name = filename.rsplit('.', 1)[0]
        s3.upload_file(filepath,"meowyeokbucket",f"comics/Pages/{image_name}.jpg")
        
        # 이미지를 넣어서 결과와, DB에 저장할 thread_id를 얻어옴
        res, thread_id = get_result(image_name, is_new_assidtant=False, id=thread_id) 
        return res, 200
    else:
        return jsonify({'error': 'File type is not .jpg'}), 400

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(host='0.0.0.0', debug= True, port=5000)