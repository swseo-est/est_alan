import asyncio
import concurrent.futures
import time
from langgraph_sdk import get_client

DEPLOYMENT_URL="http://10.8.1.229:8000/"
# DEPLOYMENT_URL="http://10.8.7.4:8000/"
# DEPLOYMENT_URL="http://localhost:2024/"
# DEPLOYMENT_URL="http://localhost:8000/"

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

def run_parallel(input_payloads, num_worker=50, max_workers=5):
    """
    여러 요청을 병렬로 실행하는 함수
    
    Args:
        input_payloads: 실행할 페이로드 리스트
        num_worker: 총 실행할 요청 수 (입력이 부족하면 복제)
        max_workers: 동시 실행할 최대 워커 수
    """
    # 입력 개수가 num_worker보다 적으면 복제
    if len(input_payloads) < num_worker:
        # 필요한 만큼 복제
        multiplier = (num_worker // len(input_payloads)) + 1
        extended_payloads = input_payloads * multiplier
        # 정확한 개수만큼만 사용
        final_payloads = extended_payloads[:num_worker]
        print(f"입력 개수({len(input_payloads)})가 num_worker({num_worker})보다 적어서 {multiplier}배 복제하여 총 {len(final_payloads)}개 요청으로 실행합니다.")
    else:
        final_payloads = input_payloads[:num_worker]
        print(f"입력 개수({len(input_payloads)})가 충분하여 처음 {len(final_payloads)}개 요청으로 실행합니다.")
    
    print(f"스트레스 테스트 시작: {len(final_payloads)}개 요청, 최대 {max_workers}개 동시 실행")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 요청을 제출
        future_to_payload = {
            executor.submit(run, payload): payload 
            for payload in final_payloads
        }
        
        # 완료된 요청들을 처리
        completed = 0
        for future in concurrent.futures.as_completed(future_to_payload):
            payload = future_to_payload[future]
            try:
                result = future.result()
                completed += 1
                print(f"완료: {completed}/{len(final_payloads)} - {payload}")
            except Exception as exc:
                print(f"오류 발생: {payload} - {exc}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n스트레스 테스트 완료!")
    print(f"총 실행 시간: {total_time:.2f}초")
    print(f"평균 처리 시간: {total_time/len(final_payloads):.2f}초/요청")

if __name__ == '__main__':
    # 단일 실행
    # payload = {"messages": "제주도 여행"}
    # run(payload)
    
    # 스트레스 테스트용 병렬 실행
    test_payloads = [
        {"messages": "제주도 여행"},
        {"messages": "서울 관광지 추천"},
        {"messages": "부산 맛집 소개"},
        {"messages": "강원도 자연 경관"},
        {"messages": "경주 역사 여행"},
        {"messages": "전주 한옥마을"},
        {"messages": "여수 바다 여행"},
        {"messages": "대구 도시 탐방"},
        {"messages": "인천 차이나타운"},
        {"messages": "광주 문화 여행"}
    ]
    
    # 병렬 실행 (최대 3개 동시 실행)
    run_parallel(test_payloads, num_worker=50, max_workers=1000)
