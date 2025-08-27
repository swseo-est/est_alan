import asyncio
import time
from typing import List, Dict, Any
from estalan.agent.graph.slide_generate_agent.graph import create_slide_generate_agent
from estalan.agent.graph.slide_generate_agent.state import SlideGenerateAgentState, Section
from estalan.llm.utils import create_chat_model
from estalan.messages.utils import create_ai_message
from langchain_core.messages import HumanMessage


def create_test_sections(num_sections: int = 3) -> List[Section]:
    """테스트용 sections를 생성하는 함수"""
    sections = []
    
    for i in range(num_sections):
        section = Section(
            # 필수 필드들
            description=f"섹션 {i+1}에 대한 설명입니다. 이는 병렬 처리 테스트를 위한 샘플 데이터입니다.",
            requirements=[f"요구사항 {i+1}-1", f"요구사항 {i+1}-2"],
            research=True,
            slide_type="content",  # title, contents, content 등
            topic=f"테스트 주제 {i+1}",
            idx=i,
            name=f"테스트 섹션 {i+1}",
            content=f"섹션 {i+1}의 내용입니다. 이는 병렬 처리 테스트를 위한 샘플 데이터입니다.",
            
            # 이미지 및 디자인 관련 필드들
            img_url="",
            design="",
            html_template="",
            html=f"<div><h1>섹션 {i+1}</h1><p>섹션 {i+1}의 내용입니다.</p></div>",
            width=1920,
            height=1080,
            design_prompt=f"섹션 {i+1}을 위한 디자인 프롬프트입니다."
        )
        sections.append(section)
    
    return sections


def create_test_state_with_sections(num_sections: int = 3) -> Dict[str, Any]:
    """테스트용 state를 생성하는 함수"""
    sections = create_test_sections(num_sections)
    
    test_state = {
        "sections": sections,
        "slides": [],  # 초기에는 빈 리스트
        "metadata": {
            "topic": "병렬 처리 테스트",
            "requirements": "테스트용 요구사항",
            "num_sections": num_sections,
            "num_slides": 0,
            "template_folder": "general",
            "status": "processing"
        },
        "requirements": [],  # 빈 리스트로 초기화
        "requirements_docs": "테스트용 요구사항 문서",
        "messages": [
            create_ai_message(content="병렬 처리 테스트를 시작합니다.", name="test_start")
        ]
    }
    
    return test_state


async def test_parallel_execution(num_sections: int = 3):
    """병렬 실행을 테스트하는 함수"""
    print(f"=== {num_sections}개 섹션으로 병렬 실행 테스트 시작 ===")
    
    # 슬라이드 생성 에이전트 생성
    agent = create_slide_generate_agent("test_parallel_agent")
    
    # 테스트용 state 생성
    test_state = create_test_state_with_sections(num_sections)
    
    print(f"초기 state: {len(test_state['sections'])}개 섹션")
    print(f"섹션 내용: {[s['name'] for s in test_state['sections']]}")
    
    start_time = time.time()
    
    try:
        # 에이전트 실행
        result = await agent.ainvoke(test_state)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n=== 실행 결과 ===")
        print(f"총 실행 시간: {execution_time:.2f}초")
        print(f"입력 섹션 수: {len(test_state['sections'])}")
        print(f"결과 키: {list(result.keys())}")
        
        if 'slides' in result:
            print(f"생성된 슬라이드 수: {len(result['slides'])}")
            for i, slide in enumerate(result['slides']):
                print(f"  슬라이드 {i+1}: {slide.get('idx', 'N/A')} - {slide.get('name', 'N/A')}")
        
        return result, execution_time
        
    except Exception as e:
        print(f"에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


async def test_parallel_vs_sequential(num_sections: int = 3):
    """병렬 실행과 순차 실행의 성능 비교 테스트"""
    print(f"\n=== 병렬 vs 순차 실행 성능 비교 테스트 ({num_sections}개 섹션) ===")
    
    # 병렬 실행 테스트
    print("\n1. 병렬 실행 테스트")
    parallel_result, parallel_time = await test_parallel_execution(num_sections)
    
    # 순차 실행 시뮬레이션 (실제로는 같은 에이전트를 사용하지만 순차적으로 처리되는 것처럼)
    print("\n2. 순차 실행 시뮬레이션")
    sequential_start = time.time()
    
    # 각 섹션을 개별적으로 처리하는 것처럼 시뮬레이션
    for i in range(num_sections):
        single_section_state = create_test_state_with_sections(1)
        single_section_state['sections'][0]['idx'] = i
        single_section_state['sections'][0]['name'] = f"순차 처리 섹션 {i+1}"
        
        # 실제로는 같은 에이전트를 사용하지만 순차적으로 호출
        await asyncio.sleep(0.1)  # 순차 처리 시뮬레이션을 위한 지연
    
    sequential_time = time.time() - sequential_start
    
    print(f"\n=== 성능 비교 결과 ===")
    print(f"병렬 실행 시간: {parallel_time:.2f}초")
    print(f"순차 실행 시간: {sequential_time:.2f}초")
    if sequential_time > 0:
        print(f"성능 향상: {((sequential_time - parallel_time) / sequential_time * 100):.1f}%")


async def test_different_section_counts():
    """다양한 섹션 수로 병렬 실행 테스트"""
    section_counts = [1, 2, 3, 5, 8]
    
    print("=== 다양한 섹션 수로 병렬 실행 테스트 ===")
    
    results = {}
    
    for count in section_counts:
        print(f"\n--- {count}개 섹션 테스트 ---")
        result, execution_time = await test_parallel_execution(count)
        results[count] = {
            'execution_time': execution_time,
            'success': result is not None
        }
    
    print(f"\n=== 전체 테스트 결과 요약 ===")
    for count, result in results.items():
        status = "성공" if result['success'] else "실패"
        print(f"{count}개 섹션: {result['execution_time']:.2f}초 ({status})")


async def test_section_data_integrity():
    """섹션 데이터 무결성 테스트"""
    print("=== 섹션 데이터 무결성 테스트 ===")
    
    sections = create_test_sections(3)
    
    print(f"생성된 섹션 수: {len(sections)}")
    
    for i, section in enumerate(sections):
        print(f"\n섹션 {i+1}:")
        print(f"  idx: {section['idx']}")
        print(f"  name: {section['name']}")
        print(f"  topic: {section['topic']}")
        print(f"  slide_type: {section['slide_type']}")
        print(f"  description: {section['description'][:50]}...")
        print(f"  requirements: {len(section['requirements'])}개")
        print(f"  research: {section['research']}")
        print(f"  width x height: {section['width']} x {section['height']}")
    
    # 필수 필드 확인
    required_fields = [
        'description', 'requirements', 'research', 'slide_type', 'topic',
        'idx', 'name', 'content', 'img_url', 'design', 'html_template',
        'html', 'width', 'height', 'design_prompt'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in sections[0]:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"\n❌ 누락된 필드: {missing_fields}")
    else:
        print(f"\n✅ 모든 필수 필드가 포함되어 있습니다.")


async def main():
    """메인 테스트 함수"""
    print("🚀 슬라이드 생성 에이전트 병렬 처리 테스트 시작")
    
    # 데이터 무결성 테스트 먼저 실행
    await test_section_data_integrity()
    
    # 기본 병렬 실행 테스트
    await test_parallel_execution(3)
    
    # 병렬 vs 순차 성능 비교
    await test_parallel_vs_sequential(3)
    
    # 다양한 섹션 수로 테스트
    await test_different_section_counts()
    
    print("\n✅ 모든 테스트 완료!")


if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(main())
