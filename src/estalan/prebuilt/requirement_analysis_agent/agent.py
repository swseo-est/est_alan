from estalan.llm.utils import create_chat_model

from estalan.prebuilt.react_agent import create_react_agent
from estalan.prebuilt.requirement_analysis_agent.state import RequirementCollectionAgentState, RequirementCollectionState
from estalan.prebuilt.requirement_analysis_agent.tools import Tools
from estalan.prebuilt.requirement_analysis_agent.context_schema import Configuration
from estalan.prebuilt.requirement_analysis_agent.prompt import PROMPT_REQUIREMENT_ANALYSIS
from estalan.agent.base.state import state_to_json_pretty, state_to_json_compact
from estalan.prebuilt.requirement_analysis_agent.converter import requirements_to_markdown, markdown_to_requirements, validate_conversion
import json


def post_agent_node(state):
    requirements = state.get('requirements', [])
    
    # JSON과 Markdown 모두 생성
    json_docs = state_to_json_compact(requirements)
    markdown_docs = requirements_to_markdown(requirements)
    
    return {
        "requirements_docs": markdown_docs,  # Markdown 형태로 변경
        "requirements_json": json_docs       # JSON 형태도 유지
    }


def create_requirement_analysis_agent(model=None, name="requirement_analysis_agent"):
    if model is None:
        model = create_chat_model(provider="azure_openai", model="gpt-4o")

    agent = create_react_agent(
        model=model,
        tools=Tools,
        prompt=PROMPT_REQUIREMENT_ANALYSIS,
        state_schema=RequirementCollectionAgentState,
        response_format=RequirementCollectionState,
        post_agent_node=post_agent_node,
        name=name
    )
    return agent


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # 테스트
    test_requirements = [
        {
            'requirement_id': 'req_001', 
            'category': '기능적', 
            'detail': '투자 IR 자료 생성', 
            'priority': 'High', 
            'status': 'draft', 
            'impact': ['투자자', '경영진'], 
            'origin': 'user'
        },
        {
            'requirement_id': 'req_002', 
            'category': '비기능적', 
            'detail': '5개 섹션으로 구성', 
            'priority': 'Medium', 
            'status': 'draft', 
            'impact': ['사용자'], 
            'origin': 'user'
        }
    ]
    
    print("=== JSON -> Markdown 변환 테스트 ===")
    markdown = requirements_to_markdown(test_requirements)
    print(markdown)
    
    print("\n=== Markdown -> JSON 변환 테스트 ===")
    converted_requirements = markdown_to_requirements(markdown)
    print(json.dumps(converted_requirements, ensure_ascii=False, indent=2))
    
    print("\n=== 변환 검증 테스트 ===")
    is_valid = validate_conversion(test_requirements)
    print(f"변환이 정확한가?: {is_valid}")
    
    print("\n=== 원본과 변환 결과 비교 ===")
    print("원본:", test_requirements)
    print("변환:", converted_requirements)
    print("일치:", test_requirements == converted_requirements)