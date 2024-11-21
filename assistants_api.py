import os
from openai import OpenAI
import time
import requests
from dotenv import load_dotenv

load_dotenv()
openai_key = os.environ.get("openai_key")

API_KEY = openai_key
client = OpenAI(api_key = API_KEY)


def new_assistant(): # assistant 생성
    assistant = client.beta.assistants.create(
    name="comics describer",
    response_format={ "type": "json_object" },
    instructions="""
1.Convert the following texts into a script format as dialogue between characters. 
2.You must replace "unknown" with appropriate names in dialogue field. 
3.You must not create any extra dialogue where it doesn't exist. If there is no dialogue, leave it. 
4.Briefly describe the scene in the picture in two lines based on the provided dialogue. 
5.Translate only the description and names into Korean.

Expected JSON structure:
[
  {
    "description": "보라색 머리의 사람과 파란 강아지가 반짝이는 큰 눈으로 기대에 찬 표정을 하고 있다.",
    "dialogue": [
      {"보라색 머리의 사람": "...sort and bag back issues, dust, clean, and maybe even do some light construction."},
      {"파란 강아지": "And listen to their unfunny stories!"}
    ]
  },
  {
    "description": "...",
    "dialogue": [
      {"unknown": "[No dialogue]"}
    ]
  }
]

    """,
    model="gpt-4o"
    )
    return assistant.id




def new_book(): #책마다 최초1회 부여해놓고 생성된 thread_id를 db에 저장해놓고 이어 읽을때 사용
    thread = client.beta.threads.create()
    return thread.id


def fetch_image_url(name, index):
    return f"https://meowyeokbucket.s3.ap-northeast-2.amazonaws.com/comics/panel_seg/{name}/{name}_{index}.jpg"

def image_exists(image_url): # 이미지 있나 확인
    response = requests.head(image_url)
    return response.status_code == 200

def create_message_content(image_url, texts_str): # assistant api에 넣어줄 메세지를 생성(image_url, text)
    content = [{
        "type": "image_url",
        "image_url": {"url": image_url, "detail": "low"}
    }]
    
    if texts_str.strip():  # 문자열 비어있나 확인
        content.append({
            "type": "text",
            "text": texts_str  # 문자열 추가
        })

    return content

def wait_for_run_completion(run): # run을 해놓고 끝났는지 주기적으로 확인해 줘야함
    while run.status != 'completed':
        time.sleep(0.1)  # 확인 빈도
        run = client.beta.threads.runs.retrieve(
            thread_id=run.thread_id,
            run_id=run.id
        )
        if run.status == 'failed':
            print("Run failed with error:", run.last_error)
            return None 

    return run


def collect_messages(messages, target_run_id):
    # run_id랑 target_run_id가 같은 애들의 메시지만 모은다.
    # run의 응답을 모두 받아버리면 과거의 받은 내용까지 모두 받아져서 중복이 계속 발생한다. 그래서 해당 run에 대한 응답만 파싱해서 리턴한다. (순서 때문에)
    collected_messages = [] 
    for message in messages:
        role = message.role
        content_list = message.content
        if message.run_id == target_run_id and role == 'assistant': # assistant의 응답중 id값이 같은지 확인
            message_texts = []  
            for content in content_list:
                if hasattr(content, 'text'):  # text값이 있으면 추가한다.
                    message_texts.append(content.text.value)
            if message_texts:
                collected_messages.append(message_texts)  # 위의 내용을 반복해서 2차원 배열로 모아 리턴
    return collected_messages


def assistant_image_captioning(name, texts, assistant_id, thread_id): # 위의 함수들을 모두 이용한 assistant api 사용 함수
    all_collected_messages = [] 

    image_index = 0
    while True: 
        image_url = fetch_image_url(name, image_index)

        if not image_exists(image_url): 
            break

        texts_str = ",".join(["".join(text) for text in texts[image_index]]) if image_index < len(texts) else ""

        content = create_message_content(image_url, texts_str)

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id 
        )

        run = wait_for_run_completion(run)
        if run is None:  
            break
        
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            collected_messages = collect_messages(messages.data, run.id)  
            all_collected_messages.extend(collected_messages) 
        
        image_index += 1

    return all_collected_messages  # gpt api의 결과인 대본 형식의 모든 메시지들 리턴

