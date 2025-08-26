# 요구사항 분석 에이전트 생성 모듈
# 사용자의 자연어 입력을 분석하여 구체적인 요구사항으로 변환하는 에이전트를 생성합니다.

from estalan.llm.utils import create_chat_model

from estalan.prebuilt.react_agent import create_react_agent
from estalan.prebuilt.requirement_analysis_agent.state import RequirementCollectionAgentState, RequirementCollectionState
from estalan.prebuilt.requirement_analysis_agent.tools import Tools
from estalan.prebuilt.requirement_analysis_agent.context_schema import Configuration
from estalan.prebuilt.requirement_analysis_agent.prompt import PROMPT_REQUIREMENT_ANALYSIS
from estalan.agent.base.state import state_to_json_pretty, state_to_json_compact


def post_agent_node(state):
    """
    에이전트 실행 후 처리 노드
    요구사항 목록을 JSON 형태로 변환하여 문서화합니다.
    
    Args:
        state: 에이전트 상태 정보
        
    Returns:
        dict: 요구사항 문서화 결과
    """
    requirements = state.get('requirements', [])
    docs = state_to_json_compact(requirements)
    return {"requirements_docs": docs}


def create_requirement_analysis_agent(model=None, name="requirement_analysis_agent"):
    """
    요구사항 분석 에이전트를 생성합니다.
    
    Args:
        model: 사용할 언어 모델 (기본값: Azure OpenAI GPT-4o)
        name: 에이전트 이름 (기본값: "requirement_analysis_agent")
        
    Returns:
        생성된 요구사항 분석 에이전트
    """
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
    # 테스트 실행 코드
    from dotenv import load_dotenv

    load_dotenv()

    graph = create_requirement_analysis_agent()
    result = graph.invoke({"messages": "나는 3박 4일 제주도 여행을 계획 중이야."})
    print(result)