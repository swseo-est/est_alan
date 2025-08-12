from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Union, TypedDict
from langchain_core.language_models import BaseLanguageModel
from langgraph_supervisor import create_supervisor
from langchain_core.messages import AIMessage

from estalan.agent.graph.requirement_collection_agent.state import RequirementCollectionAgentState, RequirementCollectionAgentPrivateState
from estalan.agent.graph.requirement_collection_agent.question_generation_agent import create_question_generation_agent
from estalan.agent.graph.requirement_collection_agent.requirement_analysis_agent import create_requirement_analysis_agent
from estalan.agent.base.node import create_initialization_node
from estalan.llm.utils import create_chat_model


def create_final_confirmation_agent(name="final_confirmation_agent", private_state_key="requirement_collection_agent_state"):
    """요구사항 수집 완료 후 최종 확인 에이전트"""
    
    def create_final_confirmation_node():
        """최종 확인 노드 생성"""
        
        def final_confirmation_node(state: RequirementCollectionAgentState):
            private_state = state.get(private_state_key, {})
            requirements = private_state.get("requirements", "")
            purpose = private_state.get("purpose", "")
            
            if not requirements or requirements == "## 요구사항\n\n수집된 요구사항이 없습니다.":
                msg = "## ⚠️ 요구사항 수집 실패\n\n요구사항을 수집하지 못했습니다.\n\n다시 시도해주시거나 다른 방법으로 요구사항을 전달해주세요."
            else:
                msg = f"""## 🎯 요구사항 수집 완료!

{purpose}를 위한 요구사항 수집이 완료되었습니다.

---

{requirements}

---

## ✅ 다음 단계

위의 요구사항으로 작업을 진행하시겠습니까?

**확인하신 후 다음 중 하나를 선택해주세요:**
1. **진행**: 요구사항이 맞습니다. 작업을 시작해주세요.
2. **수정**: 일부 요구사항을 수정하고 싶습니다.
3. **추가**: 더 많은 요구사항이 있습니다.
4. **취소**: 작업을 취소하고 싶습니다.

어떤 선택을 하시겠습니까?"""

            return {"messages": [AIMessage(content=msg)]}
        
        return final_confirmation_node

    # 그래프 구성
    builder = StateGraph(RequirementCollectionAgentState)
    
    # 노드 추가
    final_confirmation_node = create_final_confirmation_node()
    builder.add_node("final_confirmation_node", final_confirmation_node)
    
    # 엣지 추가
    builder.add_edge(START, "final_confirmation_node")
    builder.add_edge("final_confirmation_node", END)
    
    # 에이전트 컴파일
    final_confirmation_agent = builder.compile(name=name)
    return final_confirmation_agent


def create_requirement_collection_agent(
        purpose=None,
        predefined_questions=[],
        name="requirement_collection_agent",
        state_schema=RequirementCollectionAgentState,
        private_state_key="requirement_collection_agent_state"
        ):
   
   # 노드 생성
   initialization_node = create_initialization_node(
        purpose=purpose,
        new_questions=predefined_questions,
        private_state_key=private_state_key
   )

   requirement_analysis_agent = create_requirement_analysis_agent(name="requirement_analysis_agent", private_state_key=private_state_key)
   question_generation_agent = create_question_generation_agent(name="question_generation_agent", private_state_key=private_state_key)
   final_confirmation_agent = create_final_confirmation_agent(name="final_confirmation_agent", private_state_key=private_state_key)

   # langgraph_supervisor를 사용한 supervisor 생성
   supervisor_prompt = f"""
   당신은 {purpose}를 위한 요구사항 수집 워크플로우의 supervisor입니다. 
   당신의 역할은 오직 적절한 에이전트를 선택하여 라우팅하는 것입니다.
   ** 최대한 사용자 메시지를 하위 에이전트로 라우팅하는 것이 중요합니다.**

   ## 에이전트 역할

   1. **requirement_analysis_agent**: 사용자의 답변에서 요구사항을 추출하고 분석
   - 사용자 메시지에서 명시적/암묵적 요구사항 추출
   - 기존 요구사항과의 연관성 분석
   - 요구사항 카테고리 분류 및 상세 내용 정리
   - 기존 요구사항 업데이트 또는 새로운 요구사항 추가

   2. **question_generation_agent**: 사용자에게 질문을 생성하여 요구사항 수집을 완성
   - 기존 질문과 중복되지 않는 새로운 질문 생성
   - 아직 파악되지 않은 요구사항 영역에 대한 질문 생성
   - 목적에 맞는 구체적이고 명확한 질문 생성
   - **유일하게 사용자에게 질문을 하는 에이전트**

   3. **final_confirmation_agent**: 요구사항 수집 완료 후 최종 확인
   - 수집된 모든 요구사항을 정리하여 표시
   - 작업 진행 전 사용자 확인 요청
   - 다음 단계 선택 안내

   ## 호출 순서
    - 질문이 더 필요한 경우 question_generation_agent 호출
    - 질문이 더 필요하지 않은 경우 final_confirmation_agent 호출

   ## 출력
   - 직전 메시지가 AIMessage인 경우 "" 출력 
   - {private_state_key}.last_step 을 가장 마지막에 호출한 agent name으로 업데이트 하세요
   """

   llm = create_chat_model(provider="azure_openai", model="gpt-4.1")


   # langgraph_supervisor를 사용하여 supervisor 생성
   supervisor_agent = create_supervisor(
   agents=[requirement_analysis_agent, question_generation_agent, final_confirmation_agent],
   model=llm,
   prompt=supervisor_prompt,
   state_schema=state_schema,
   output_mode="full_history"
   ).compile()

   # 그래프 구성
   builder = StateGraph(state_schema)

   # 노드 추가
   builder.add_node("initialization_node", initialization_node)
   builder.add_node("requirement_analysis_agent", requirement_analysis_agent)

   builder.add_node("supervisor_agent", supervisor_agent)

   builder.add_edge(START, "initialization_node")
   builder.add_edge("initialization_node", "requirement_analysis_agent")
   builder.add_edge("requirement_analysis_agent", "supervisor_agent")
   builder.add_edge("supervisor_agent", END)

   # 그래프 컴파일
   graph = builder.compile(name=name)

   return graph
