import pytest
import json
from typing import List, Dict, Any

from estalan.prebuilt.requirement_analysis_agent.converter import (
    requirements_to_markdown,
    markdown_to_requirements,
    validate_conversion
)


def print_conversion_info(original_data, converted_data, title):
    """변환 정보를 출력하는 헬퍼 함수"""
    print(f"\n=== {title} ===")
    print(f"원본 데이터: {json.dumps(original_data, ensure_ascii=False, indent=2)}")
    print(f"변환된 데이터: {json.dumps(converted_data, ensure_ascii=False, indent=2)}")


def test_requirements_to_markdown_basic():
    """기본 요구사항을 Markdown으로 변환하는 테스트"""
    requirements = [
        {
            'requirement_id': 'req_001',
            'category': '기능적',
            'detail': '투자 IR 자료 생성',
            'priority': 'High',
            'status': 'draft',
            'impact': ['투자자', '경영진'],
            'origin': 'user'
        }
    ]
    
    markdown = requirements_to_markdown(requirements)
    
    # 기본 구조 검증
    assert "# 사용자 요구사항" in markdown
    assert "## 요구사항 1" in markdown
    assert "**ID**: req_001" in markdown
    assert "**카테고리**: 기능적" in markdown
    assert "**상세**: 투자 IR 자료 생성" in markdown
    assert "**우선순위**: High" in markdown
    assert "**상태**: draft" in markdown
    assert "**영향받는 대상**: 투자자, 경영진" in markdown
    assert "**출처**: user" in markdown


def test_requirements_to_markdown_multiple():
    """여러 요구사항을 Markdown으로 변환하는 테스트"""
    requirements = [
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
    
    markdown = requirements_to_markdown(requirements)
    print(markdown)
    
    # 여러 요구사항 검증
    assert markdown.count("## 요구사항") == 2
    assert "**ID**: req_001" in markdown
    assert "**ID**: req_002" in markdown
    assert "**카테고리**: 기능적" in markdown
    assert "**카테고리**: 비기능적" in markdown


def test_requirements_to_markdown_empty_impact():
    """영향받는 대상이 없는 요구사항 변환 테스트"""
    requirements = [
        {
            'requirement_id': 'req_001',
            'category': '기능적',
            'detail': '테스트 요구사항',
            'priority': 'Low',
            'status': 'draft',
            'impact': [],
            'origin': 'user'
        }
    ]
    
    markdown = requirements_to_markdown(requirements)
    
    assert "**영향받는 대상**: 없음" in markdown


def test_requirements_to_markdown_missing_fields():
    """일부 필드가 누락된 요구사항 변환 테스트"""
    requirements = [
        {
            'requirement_id': 'req_001',
            'category': '기능적',
            'detail': '테스트 요구사항',
            # priority, status, impact, origin 필드 누락
        }
    ]
    
    markdown = requirements_to_markdown(requirements)
    
    assert "**우선순위**: N/A" in markdown
    assert "**상태**: N/A" in markdown
    assert "**영향받는 대상**: 없음" in markdown
    assert "**출처**: N/A" in markdown


def test_requirements_to_markdown_empty_list():
    """빈 요구사항 리스트 변환 테스트"""
    requirements = []
    
    markdown = requirements_to_markdown(requirements)
    
    assert markdown == "사용자 요구사항이 없습니다."


def test_markdown_to_requirements_basic():
    """기본 Markdown을 요구사항으로 변환하는 테스트"""
    markdown = """# 사용자 요구사항

## 요구사항 1
- **ID**: req_001
- **카테고리**: 기능적
- **상세**: 투자 IR 자료 생성
- **우선순위**: High
- **상태**: draft
- **영향받는 대상**: 투자자, 경영진
- **출처**: user
"""
    
    requirements = markdown_to_requirements(markdown)
    
    assert len(requirements) == 1
    req = requirements[0]
    assert req['requirement_id'] == 'req_001'
    assert req['category'] == '기능적'
    assert req['detail'] == '투자 IR 자료 생성'
    assert req['priority'] == 'High'
    assert req['status'] == 'draft'
    assert req['impact'] == ['투자자', '경영진']
    assert req['origin'] == 'user'


def test_markdown_to_requirements_multiple():
    """여러 요구사항이 포함된 Markdown 변환 테스트"""
    markdown = """# 사용자 요구사항

## 요구사항 1
- **ID**: req_001
- **카테고리**: 기능적
- **상세**: 투자 IR 자료 생성
- **우선순위**: High
- **상태**: draft
- **영향받는 대상**: 투자자, 경영진
- **출처**: user

## 요구사항 2
- **ID**: req_002
- **카테고리**: 비기능적
- **상세**: 5개 섹션으로 구성
- **우선순위**: Medium
- **상태**: draft
- **영향받는 대상**: 사용자
- **출처**: user
"""
    
    requirements = markdown_to_requirements(markdown)
    
    assert len(requirements) == 2
    
    # 첫 번째 요구사항 검증
    req1 = requirements[0]
    assert req1['requirement_id'] == 'req_001'
    assert req1['category'] == '기능적'
    
    # 두 번째 요구사항 검증
    req2 = requirements[1]
    assert req2['requirement_id'] == 'req_002'
    assert req2['category'] == '비기능적'
    assert req2['impact'] == ['사용자']


def test_markdown_to_requirements_empty_impact():
    """영향받는 대상이 없는 Markdown 변환 테스트"""
    markdown = """# 사용자 요구사항

## 요구사항 1
- **ID**: req_001
- **카테고리**: 기능적
- **상세**: 테스트 요구사항
- **우선순위**: Low
- **상태**: draft
- **영향받는 대상**: 없음
- **출처**: user
"""
    
    requirements = markdown_to_requirements(markdown)
    
    assert len(requirements) == 1
    req = requirements[0]
    assert req['impact'] == []


def test_markdown_to_requirements_missing_fields():
    """일부 필드가 누락된 Markdown 변환 테스트"""
    markdown = """# 사용자 요구사항

## 요구사항 1
- **ID**: req_001
- **카테고리**: 기능적
- **상세**: 테스트 요구사항
"""
    
    requirements = markdown_to_requirements(markdown)
    
    assert len(requirements) == 1
    req = requirements[0]
    assert req['requirement_id'] == 'req_001'
    assert req['category'] == '기능적'
    assert req['detail'] == '테스트 요구사항'
    # 누락된 필드들은 빈 딕셔너리로 처리됨


def test_markdown_to_requirements_empty_markdown():
    """빈 Markdown 변환 테스트"""
    markdown = ""
    requirements = markdown_to_requirements(markdown)
    assert requirements == []


def test_markdown_to_requirements_no_requirements():
    """요구사항이 없다는 메시지가 포함된 Markdown 변환 테스트"""
    markdown = "사용자 요구사항이 없습니다."
    requirements = markdown_to_requirements(markdown)
    assert requirements == []


def test_markdown_to_requirements_malformed():
    """잘못된 형식의 Markdown 변환 테스트"""
    markdown = """# 사용자 요구사항

잘못된 형식의 마크다운
- 필드명이 없음
"""
    
    requirements = markdown_to_requirements(markdown)
    assert requirements == []


def test_validate_conversion_success():
    """변환 검증 성공 테스트"""
    original_requirements = [
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
    
    is_valid = validate_conversion(original_requirements)
    assert is_valid is True


def test_validate_conversion_empty_list():
    """빈 리스트 변환 검증 테스트"""
    original_requirements = []
    
    is_valid = validate_conversion(original_requirements)
    assert is_valid is True


def test_validate_conversion_complex_impact():
    """복잡한 impact 필드 변환 검증 테스트"""
    original_requirements = [
        {
            'requirement_id': 'req_001',
            'category': '기능적',
            'detail': '복잡한 요구사항',
            'priority': 'High',
            'status': 'draft',
            'impact': ['사용자', '관리자', '개발자', '테스터'],
            'origin': 'user'
        }
    ]
    
    is_valid = validate_conversion(original_requirements)
    assert is_valid is True


def test_round_trip_conversion():
    """JSON -> Markdown -> JSON 순환 변환 테스트"""
    original_requirements = [
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
    
    # JSON -> Markdown
    markdown = requirements_to_markdown(original_requirements)
    
    # Markdown -> JSON
    converted_requirements = markdown_to_requirements(markdown)
    
    # 결과 출력
    print_conversion_info(original_requirements, converted_requirements, "순환 변환 테스트")
    
    # 검증
    assert len(original_requirements) == len(converted_requirements)
    
    for i, (original, converted) in enumerate(zip(original_requirements, converted_requirements)):
        assert original['requirement_id'] == converted['requirement_id']
        assert original['category'] == converted['category']
        assert original['detail'] == converted['detail']
        assert original['priority'] == converted['priority']
        assert original['status'] == converted['status']
        assert original['impact'] == converted['impact']
        assert original['origin'] == converted['origin']


def test_conversion_with_special_characters():
    """특수 문자가 포함된 요구사항 변환 테스트"""
    original_requirements = [
        {
            'requirement_id': 'req_001',
            'category': '기능적',
            'detail': '특수문자 테스트: !@#$%^&*()_+-=[]{}|;:,.<>?',
            'priority': 'High',
            'status': 'draft',
            'impact': ['사용자', '관리자'],
            'origin': 'user'
        }
    ]
    
    markdown = requirements_to_markdown(original_requirements)
    converted_requirements = markdown_to_requirements(markdown)
    
    assert len(converted_requirements) == 1
    assert converted_requirements[0]['detail'] == original_requirements[0]['detail']


def test_conversion_with_multiline_detail():
    """여러 줄 상세 내용 변환 테스트"""
    original_requirements = [
        {
            'requirement_id': 'req_001',
            'category': '기능적',
            'detail': '첫 번째 줄\n두 번째 줄\n세 번째 줄',
            'priority': 'High',
            'status': 'draft',
            'impact': ['사용자'],
            'origin': 'user'
        }
    ]
    
    markdown = requirements_to_markdown(original_requirements)
    converted_requirements = markdown_to_requirements(markdown)
    
    assert len(converted_requirements) == 1
    # 여러 줄 텍스트는 줄바꿈이 공백으로 변환됨
    expected_detail = '첫 번째 줄 두 번째 줄 세 번째 줄'
    assert converted_requirements[0]['detail'] == expected_detail


def test_conversion_with_empty_strings():
    """빈 문자열이 포함된 요구사항 변환 테스트"""
    original_requirements = [
        {
            'requirement_id': '',
            'category': '',
            'detail': '',
            'priority': '',
            'status': '',
            'impact': [],
            'origin': ''
        }
    ]
    
    markdown = requirements_to_markdown(original_requirements)
    converted_requirements = markdown_to_requirements(markdown)
    
    assert len(converted_requirements) == 1
    req = converted_requirements[0]
    # 빈 문자열은 'N/A'로 변환됨
    assert req['requirement_id'] == 'N/A'
    assert req['category'] == 'N/A'
    assert req['detail'] == 'N/A'
    assert req['priority'] == 'N/A'
    assert req['status'] == 'N/A'
    assert req['impact'] == []
    assert req['origin'] == 'N/A'


def test_conversion_with_none_values():
    """None 값이 포함된 요구사항 변환 테스트"""
    original_requirements = [
        {
            'requirement_id': None,
            'category': None,
            'detail': None,
            'priority': None,
            'status': None,
            'impact': None,
            'origin': None
        }
    ]
    
    markdown = requirements_to_markdown(original_requirements)
    converted_requirements = markdown_to_requirements(markdown)
    
    assert len(converted_requirements) == 1
    req = converted_requirements[0]
    assert req['requirement_id'] == 'N/A'
    assert req['category'] == 'N/A'
    assert req['detail'] == 'N/A'
    assert req['priority'] == 'N/A'
    assert req['status'] == 'N/A'
    assert req['impact'] == []
    assert req['origin'] == 'N/A'


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v"])
