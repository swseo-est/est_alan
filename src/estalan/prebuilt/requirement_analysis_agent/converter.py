"""
요구사항 JSON과 Markdown 간의 상호변환 모듈
"""

import json
import re
from typing import List, Dict, Any


def requirements_to_markdown(requirements: List[Dict[str, Any]]) -> str:
    """
    요구사항 리스트를 Markdown 형태로 변환합니다.
    
    Markdown 구조:
    ```markdown
    # 사용자 요구사항
    
    ## 요구사항 1
    - **ID**: req_001
    - **카테고리**: 기능적
    - **상세**: 투자 IR 자료 생성
    - **우선순위**: High
    - **상태**: draft
    - **영향받는 대상**: 투자자, 경영진
    - **출처**: user
    
    ## 요구사항 2
    ...
    ```
    
    Args:
        requirements: 요구사항 리스트
        
    Returns:
        Markdown 형태의 문자열
    """
    if not requirements:
        return "사용자 요구사항이 없습니다."
    
    markdown = "# 사용자 요구사항\n\n"
    
    for i, req in enumerate(requirements, 1):
        markdown += f"## 요구사항 {i}\n"
        
        # None 값과 빈 문자열 처리
        requirement_id = req.get('requirement_id', 'N/A')
        if requirement_id is None or requirement_id == '':
            requirement_id = 'N/A'
        markdown += f"- **ID**: {requirement_id}\n"
        
        category = req.get('category', 'N/A')
        if category is None or category == '':
            category = 'N/A'
        markdown += f"- **카테고리**: {category}\n"
        
        detail = req.get('detail', 'N/A')
        if detail is None or detail == '':
            detail = 'N/A'
        # 여러 줄 텍스트를 단일 줄로 변환 (줄바꿈을 공백으로)
        detail = detail.replace('\n', ' ')
        markdown += f"- **상세**: {detail}\n"
        
        priority = req.get('priority', 'N/A')
        if priority is None or priority == '':
            priority = 'N/A'
        markdown += f"- **우선순위**: {priority}\n"
        
        status = req.get('status', 'N/A')
        if status is None or status == '':
            status = 'N/A'
        markdown += f"- **상태**: {status}\n"
        
        impact = req.get('impact', [])
        if impact is None:
            impact = []
        if impact:
            markdown += f"- **영향받는 대상**: {', '.join(impact)}\n"
        else:
            markdown += "- **영향받는 대상**: 없음\n"
            
        origin = req.get('origin', 'N/A')
        if origin is None or origin == '':
            origin = 'N/A'
        markdown += f"- **출처**: {origin}\n\n"
    
    return markdown


def markdown_to_requirements(markdown: str) -> List[Dict[str, Any]]:
    """
    Markdown 형태를 요구사항 리스트로 변환합니다.
    
    Args:
        markdown: Markdown 형태의 요구사항 문서
        
    Returns:
        요구사항 리스트
    """
    if not markdown or "사용자 요구사항이 없습니다" in markdown:
        return []
    
    requirements = []
    
    # 요구사항 섹션들을 분리
    sections = re.split(r'## 요구사항 \d+', markdown)
    
    for section in sections[1:]:  # 첫 번째는 빈 문자열이므로 제외
        if not section.strip():
            continue
            
        req = {}
        
        # 각 필드 추출
        id_match = re.search(r'\*\*ID\*\*: (.+)', section)
        if id_match:
            req['requirement_id'] = id_match.group(1).strip()
        
        category_match = re.search(r'\*\*카테고리\*\*: (.+)', section)
        if category_match:
            req['category'] = category_match.group(1).strip()
        
        detail_match = re.search(r'\*\*상세\*\*: (.+)', section)
        if detail_match:
            req['detail'] = detail_match.group(1).strip()
        
        priority_match = re.search(r'\*\*우선순위\*\*: (.+)', section)
        if priority_match:
            req['priority'] = priority_match.group(1).strip()
        
        status_match = re.search(r'\*\*상태\*\*: (.+)', section)
        if status_match:
            req['status'] = status_match.group(1).strip()
        
        impact_match = re.search(r'\*\*영향받는 대상\*\*: (.+)', section)
        if impact_match:
            impact_str = impact_match.group(1).strip()
            if impact_str != "없음":
                req['impact'] = [item.strip() for item in impact_str.split(',')]
            else:
                req['impact'] = []
        
        origin_match = re.search(r'\*\*출처\*\*: (.+)', section)
        if origin_match:
            req['origin'] = origin_match.group(1).strip()
        
        if req:  # 빈 딕셔너리가 아닌 경우만 추가
            requirements.append(req)
    
    return requirements


def validate_conversion(original_requirements: List[Dict[str, Any]]) -> bool:
    """
    변환의 정확성을 검증합니다.
    
    Args:
        original_requirements: 원본 요구사항 리스트
        
    Returns:
        변환이 정확한지 여부
    """
    # JSON -> Markdown -> JSON 변환 테스트
    markdown = requirements_to_markdown(original_requirements)
    converted_requirements = markdown_to_requirements(markdown)
    
    # 필수 필드들만 비교 (순서는 무시)
    def normalize_requirement(req):
        return {
            'requirement_id': req.get('requirement_id', ''),
            'category': req.get('category', ''),
            'detail': req.get('detail', ''),
            'priority': req.get('priority', ''),
            'status': req.get('status', ''),
            'impact': sorted(req.get('impact', [])),
            'origin': req.get('origin', '')
        }
    
    original_normalized = [normalize_requirement(req) for req in original_requirements]
    converted_normalized = [normalize_requirement(req) for req in converted_requirements]
    
    return sorted(original_normalized, key=lambda x: x['requirement_id']) == sorted(converted_normalized, key=lambda x: x['requirement_id'])
