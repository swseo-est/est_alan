from langgraph.graph import StateGraph, START, END

from estalan.llm.utils import create_chat_model
from estalan.logging.base import get_logger

from estalan.prebuilt.react_agent import create_react_agent
from estalan.prebuilt.requirement_analysis_agent.state import RequirementCollectionAgentState, RequirementCollectionState
from estalan.prebuilt.requirement_analysis_agent.tools import Tools
from estalan.prebuilt.requirement_analysis_agent.prompt import PROMPT_REQUIREMENT_ANALYSIS
from estalan.prebuilt.requirement_analysis_agent.converter import requirements_to_markdown
from estalan.messages.utils import create_ai_message

# 로거 초기화
logger = get_logger(__name__)


def pre_agent_node(state):
    logger.debug("pre_agent_node 실행 시작", state_keys=list(state.keys()))
    
    requirements_docs = state.get("requirements_docs", "")

    if requirements_docs:
        msg = "기존 등록된 요구사항은 다음과 같습니다. 새로운 유저 메시지를 분석해서 요구사항을 추가/업데이트 하세요\n"
        msg += requirements_docs
        logger.info("기존 요구사항 문서 로드", requirements_count=len(requirements_docs.split('\n')))
    else:
        msg = "현재 등록된 요구사항은 없습니다."
        logger.info("새로운 요구사항 분석 시작 - 기존 요구사항 없음")

    ai_msg = create_ai_message(content=msg, name="previous_req", metadata={"log_level": "debug"})
    last_msg = state["messages"][-1]
    logger.debug("pre_agent_node 완료", message_count=1)
    return {"messages": [ai_msg]}


def post_agent_node(state):
    """
    에이전트 실행 후 처리 노드
    요구사항 목록을 JSON 형태로 변환하여 문서화합니다.

    Args:
        state: 에이전트 상태 정보

    Returns:
        dict: 요구사항 문서화 결과
    """
    logger.debug("post_agent_node 실행 시작", state_keys=list(state.keys()))
    
    requirements = state.get('requirements', [])
    logger.info("요구사항 처리 완료", requirements_count=len(requirements))
    
    # JSON과 Markdown 생성
    markdown_docs = requirements_to_markdown(requirements)
    logger.debug("Markdown 문서 생성 완료", doc_length=len(markdown_docs))

    msg = "업데이트된 요구사항은 다음과 같습니다.\n"
    msg += markdown_docs

    ai_msg = create_ai_message(content=msg, name="updated_req", metadata={"log_level": "debug"})
    logger.debug("post_agent_node 완료", message_count=1, requirements_docs_length=len(markdown_docs))

    return {
        "messages": [ai_msg],
        "requirements_docs": markdown_docs,  # Markdown 형태
    }

def pre_model_hook(state):
    logger.debug("pre_model_hook 실행", state_keys=list(state.keys()))
    return {}

def post_model_hook(state):
    logger.debug("post_model_hook 실행", state_keys=list(state.keys()))
    return {}

def create_requirement_analysis_agent(model=None, name="requirement_analysis_agent"):
    """
    요구사항 분석 에이전트를 생성합니다.

    Args:
        model: 사용할 언어 모델 (기본값: Azure OpenAI GPT-4o)
        name: 에이전트 이름 (기본값: "requirement_analysis_agent")

    Returns:
        생성된 요구사항 분석 에이전트
    """
    logger.info("요구사항 분석 에이전트 생성 시작", agent_name=name)
    
    if model is None:
        model = create_chat_model(provider="azure_openai", model="gpt-5-nano")
        logger.debug("기본 모델 생성됨", provider="azure_openai", model="gpt-5-nano")

    agent = create_react_agent(
        model=model,
        tools=Tools,
        prompt=PROMPT_REQUIREMENT_ANALYSIS,
        state_schema=RequirementCollectionAgentState,
        response_format=RequirementCollectionState,
        post_agent_node=post_agent_node,
        pre_agent_node=pre_agent_node,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        name=name
    )
    
    logger.info("요구사항 분석 에이전트 생성 완료", agent_name=name)
    return agent

def create_input_subagent_node(name=None):
    if name is None:
        raise ValueError("name is required")

    def input_subagent_node(state):
        logger.debug("input_subagent_node 실행", subagent_name=name, state_keys=list(state.keys()))
        # 외부 state에서 internal state를 추출
        internal_state = state[name]
        internal_state["messages"] = state["messages"]
        logger.debug("input_subagent_node 완료", subagent_name=name, internal_state_keys=list(internal_state.keys()))
        return internal_state
    return input_subagent_node


def create_output_subagent_node(name=None):
    if name is None:
        raise ValueError("name is required")

    def output_subagent_node(state):
        logger.debug("output_subagent_node 실행", subagent_name=name, state_keys=list(state.keys()))
        # 내부 state를 외부 state로 변환

        private_state = state
        shared_state = {"requirements_docs": state["requirements_docs"]}
        
        logger.debug("output_subagent_node 완료", subagent_name=name, 
                    private_state_keys=list(private_state.keys()),
                    shared_state_keys=list(shared_state.keys()))

        return {
            "private_state" : {name: private_state},
            "shared_state" : {name: shared_state},
        }
    return output_subagent_node


def create_requirement_analysis_subagent(model=None, state_schema=None, name="requirement_analysis_agent", state_name="requirement_analysis_agent_state"):
    logger.info("요구사항 분석 서브에이전트 생성 시작", agent_name=name, state_name=state_name)
    
    requirement_analysis_agent = create_requirement_analysis_agent(model, name)
    input_subagent_node = create_input_subagent_node(state_name)
    output_subagent_node = create_output_subagent_node(state_name)

    builder = StateGraph(state_schema)
    logger.debug("StateGraph 빌더 생성됨")

    builder.add_node("input", input_subagent_node)
    builder.add_node("output", output_subagent_node)
    builder.add_node(name, requirement_analysis_agent)
    logger.debug("노드 추가 완료", nodes=["input", "output", name])

    builder.add_edge(START, "input")
    builder.add_edge("input", name)
    builder.add_edge(name, "output")
    builder.add_edge("output", END)
    logger.debug("엣지 추가 완료")

    graph = builder.compile(name=name)
    logger.info("요구사항 분석 서브에이전트 생성 완료", agent_name=name, graph_name=name)

    return graph


if __name__ == "__main__":
    # 테스트 실행 코드
    from dotenv import load_dotenv
    from estalan.prebuilt.requirement_analysis_agent.prompt import test_prompt

    load_dotenv()
    logger.info("테스트 실행 시작")

    agent = create_requirement_analysis_agent()
    logger.info("에이전트 생성 완료, 테스트 실행 중...")
    
    result = agent.invoke({"messages": test_prompt})
    logger.info("테스트 실행 완료", result_keys=list(result.keys()))
    
    print(result["requirements_docs"])
