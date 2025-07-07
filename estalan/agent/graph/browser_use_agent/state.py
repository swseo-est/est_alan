from pydantic import BaseModel, Field
import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 공유 구조체(Shared Structures)
# ---------------------------------------------------------------------------

class BrowserSnapshot(BaseModel):
    """현재 브라우저 뷰에 대한 경량 스냅샷."""

    dom: Optional[str] = None  # 렌더링된 DOM 문자열 (필요 시 부분 샘플링)
    url: Optional[str] = None  # 현재 URL
    screenshot_path: Optional[str] = None  # 스크린샷이 저장된 로컬 경로

# ---------------------------------------------------------------------------
# Playwright MCP Tool 기반 액션 이력 구조체
# ---------------------------------------------------------------------------

class ActionHistory(BaseModel):
    tool_name: str = Field(..., description="Playwright MCP Tool 이름")
    params: Dict[str, Any] = Field(default_factory=dict, description="도구에 전달된 파라미터")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, description="액션 실행 시각(UTC)")
    result: Optional[str] = Field(default=None, description="실행 결과/상태/오류 메시지 등")

# ---------------------------------------------------------------------------
# Planner의 단기 목표 기반 Plan Step 구조
# ---------------------------------------------------------------------------

class PlanStep(BaseModel):
    """Planner가 생성하는 단기 목표 기반 step의 구조."""
    id: int = Field(..., description="step의 순서 또는 고유 식별자")
    description: str = Field(..., description="이 step에서 달성해야 할 단기 목표(예: '검색창에 김치찌개 입력 후 검색 결과 보기')")
    is_done: bool = Field(default=False, description="step 완료 여부")
    error: Optional[str] = Field(default=None, description="실패/예외 발생 시 원인 메시지")

# ---------------------------------------------------------------------------
# 내부 에이전트별 Private State
# ---------------------------------------------------------------------------

class PlannerState(BaseModel):
    """Planner 에이전트의 내부 상태."""

    goal: Optional[str] = None  # 최상위 목표
    completed_steps: List[PlanStep] = Field(default_factory=list)  # 이미 처리한 step
    remaining_steps: List[PlanStep] = Field(default_factory=list)  # 앞으로 진행할 step

class NavigatorState(BaseModel):
    """Navigator 에이전트의 내부 상태."""

    current_url: Optional[str] = None  # 현재 탐색 중인 URL
    history: List[str] = Field(default_factory=list)  # 방문 기록 (최신 우선)
    action_history: List[ActionHistory] = Field(default_factory=list)  # Playwright 액션 이력

class ValidatorState(BaseModel):
    """Validator 에이전트의 내부 상태."""

    ruleset: Dict[str, Any] = Field(default_factory=dict)  # 검증 규칙 모음
    violations: List[str] = Field(default_factory=list)  # 발견된 규칙 위반 사항

# ---------------------------------------------------------------------------
# 상위 에이전트 공유 상태
# ---------------------------------------------------------------------------

class BrowserUseAgentState(BaseModel):
    """상위 Browser‑Use‑Agent 가 보유하는 통합 상태."""

    planner: PlannerState = Field(default_factory=PlannerState)
    navigator: NavigatorState = Field(default_factory=NavigatorState)
    validator: ValidatorState = Field(default_factory=ValidatorState)

    current_plan_step: Optional[PlanStep] = None  # 현재 진행 중인 step만 공유
    snapshot: Optional[BrowserSnapshot] = None  # 최신 브라우저 스냅샷


__all__ = [
    "BrowserSnapshot",
    "ActionHistory",
    "PlanStep",
    "PlannerState",
    "NavigatorState",
    "ValidatorState",
    "BrowserUseAgentState",
]
