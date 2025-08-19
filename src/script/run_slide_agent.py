import asyncio
from langgraph_sdk import get_client

DEPLOYMENT_URL = "http://localhost:8000/"

client = get_client(url=DEPLOYMENT_URL)
list_msg_id = []

def get_data(chunk):
    if not getattr(chunk, "data", None):
        return None
    values = list(chunk.data.values())
    return values[0] if values else None

def filter_ai_msg(chunk):
    # 유저와 채팅 창에 출력될 ai 메시지를 필터링
    if 'updates' in getattr(chunk, "event", ""):
        data = get_data(chunk)
        if data and "messages" in data:
            last = data["messages"][-1]
            # name이 없을 수 있어 get 사용
            if last.get("name") is not None:
                msg_type = last.get("type")
                msg_id = last.get("id")
                if msg_id and msg_type == "ai" and msg_id not in list_msg_id:
                    list_msg_id.append(msg_id)
                    return True
    return False

def filter_slide_data(chunk):
    # slide 관련 데이터를 담은 메시지를 필터링 (오타 수정)
    if 'updates' in getattr(chunk, "event", ""):
        node_names = chunk.data.keys() if getattr(chunk, "data", None) else []
        return 'executor' in node_names
    return False

assistant_id = "slide_generate_agent"
list_chunk = []

async def call(input_payload):
    # 스레드 생성은 async 컨텍스트 안에서!
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    async for chunk in client.runs.stream(
        thread_id,
        assistant_id=assistant_id,
        input=input_payload,
        stream_mode=["updates"],
        stream_subgraphs=True
    ):
        list_chunk.append(chunk)
        if filter_ai_msg(chunk):
            print("#" * 10 + "  AI 메시지  " + "#" * 10)
            data = get_data(chunk)
            if data and "messages" in data:
                print(data["messages"][-1]["content"])
        # elif filter_slide_data(chunk):
        #     print("#" * 10 + "  슬라이드 데이터  " + "#" * 10)
        #     print(chunk)

def run(input_payload):
    # 실행용 동기 래퍼 (테스트에서는 사용 금지)
    asyncio.run(call(input_payload))

if __name__ == '__main__':
    payload = {"messages": "제주도 여행"}
    run(payload)
