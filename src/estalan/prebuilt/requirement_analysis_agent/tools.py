from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
import uuid
import json
from datetime import datetime
from .state import RequirementCollectionState, Requirement, Question, Requirement
from estalan.agent.base.state import state_to_json_pretty, state_to_json_compact


# ===== READ (조회) 도구들 =====

# ===== CREATE (생성) 도구들 =====

@tool
def create_requirement(
    category: str,
    detail: str,
    priority: str = "Medium",
    impact: Optional[List[str]] = None,
    origin: str = "user"
) -> str:
    """
    새로운 요구사항을 생성합니다.
    기존 요구사항과 중복되는지 확인하고, 중복되지 않는 경우에만 생성합니다.
    
    Args:
        category: str - 요구사항 카테고리 (기능적/비기능적/제약사항/기타)
        detail: str - 상세한 요구사항 설명
        priority: str - 우선순위 (High/Medium/Low), 기본값: Medium
        impact: Optional[List[str]] - 영향받는 시스템/프로세스/사용자 목록
        origin: str - 요구사항 출처 (user/question/inferred), 기본값: user
        
    Returns:
        str: 생성된 요구사항 정보와 결과 메시지
    """
    # 현재 상태에서 기존 요구사항들을 확인
    # (실제로는 state에서 가져와야 하지만, 도구에서는 직접 접근이 어려우므로
    # 에이전트가 중복 체크를 수행하도록 프롬프트에서 안내)
    
    requirement_id = str(uuid.uuid4())
    new_requirement: Requirement = {
        "requirement_id": requirement_id,
        "category": category,
        "detail": detail,
        "priority": priority,
        "status": "draft",  
        "impact": impact or [],
        "origin": origin
    }
    
    return state_to_json_compact(new_requirement)


# ===== UPDATE (수정) 도구들 =====

@tool
def update_requirement(
    requirement_old: Requirement,
    requirement_id: str,
    field: str,
    value: str
) -> Dict[str, Any]:
    """
    요구사항의 특정 필드를 수정합니다.
    
    Args:
        requirement_old: Requirement - 수정할 기존 요구사항
        requirement_id: str - 수정할 요구사항의 ID
        field: str - 수정할 필드명 (detail/priority/status/category)
        value: str - 새로운 값
        
    Returns:
        str: 수정된 요구사항 정보
    """
    valid_fields = {
        "detail": "상세 내용",
        "priority": "우선순위",
        "status": "상태", 
        "category": "카테고리"
    }
    
    if field not in valid_fields:
        return state_to_json_compact({
            "error": f"유효하지 않은 필드입니다. 사용 가능한 필드: {', '.join(valid_fields.keys())}"
        })

    requirement_new = requirement_old.copy()
    requirement_new[field] = value
    return state_to_json_compact(requirement_new)
    

@tool
def update_requirement_bulk(
    requirement_old: Requirement,
    requirement_id: str,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    요구사항의 여러 필드를 한번에 수정합니다.
    
    Args:
        requirement_old: Requirement - 수정할 기존 요구사항
        requirement_id: str - 수정할 요구사항의 ID
        updates: Dict[str, Any] - 수정할 필드와 값들의 딕셔너리
                예: {"detail": "새로운 내용", "priority": "High", "status": "approved"}
        
    Returns:
        str: 수정된 요구사항 정보
    """
    valid_fields = {
        "detail": "상세 내용",
        "priority": "우선순위",
        "status": "상태", 
        "category": "카테고리"
    }
    
    # 유효하지 않은 필드 확인
    invalid_fields = [field for field in updates.keys() if field not in valid_fields]
    if invalid_fields:
        return state_to_json_compact({
            "error": f"유효하지 않은 필드입니다: {', '.join(invalid_fields)}. 사용 가능한 필드: {', '.join(valid_fields.keys())}"
        })
    
    requirement_new = requirement_old.copy()
    for field, value in updates.items():
        requirement_new[field] = value
    return state_to_json_compact(requirement_new)


# ===== 도구 목록 =====

Tools = [
    # READ 도구들
    
    # CREATE 도구들
    create_requirement,
    
    # UPDATE 도구들
    update_requirement,
    update_requirement_bulk,

    # DELETE 도구들
]
