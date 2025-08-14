import os
from typing import List, Dict, Optional
from pathlib import Path
from langchain_core.tools import tool


def get_html_template_files(template_dir: str) -> List[str]:
    """
    지정된 템플릿 디렉토리에서 HTML 파일들의 리스트를 반환합니다.
    
    Args:
        template_dir (str): 템플릿 디렉토리 경로
        
    Returns:
        List[str]: HTML 파일명 리스트 (확장자 포함)
    """
    try:
        template_path = Path(template_dir)
        if not template_path.exists():
            print(f"템플릿 디렉토리가 존재하지 않습니다: {template_dir}")
            return []
        
        html_files = [f.name for f in template_path.iterdir() if f.is_file() and f.suffix.lower() == '.html']
        return sorted(html_files)
    
    except Exception as e:
        print(f"템플릿 파일 리스트를 읽는 중 오류 발생: {e}")
        return []


def get_html_template_content(template_dir: str, filename: str) -> Optional[str]:
    """
    지정된 템플릿 디렉토리에서 특정 HTML 파일의 내용을 읽어옵니다.
    
    Args:
        template_dir (str): 템플릿 디렉토리 경로
        filename (str): 읽어올 HTML 파일명
        
    Returns:
        Optional[str]: HTML 파일 내용 또는 None (파일이 없거나 읽기 실패 시)
    """
    try:
        file_path = Path(template_dir) / filename
        if not file_path.exists():
            print(f"템플릿 파일이 존재하지 않습니다: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    
    except Exception as e:
        print(f"템플릿 파일을 읽는 중 오류 발생: {e}")
        return None


def get_all_html_templates(template_dir: str) -> Dict[str, str]:
    """
    지정된 템플릿 디렉토리의 모든 HTML 파일을 읽어서 파일명과 내용을 딕셔너리로 반환합니다.
    
    Args:
        template_dir (str): 템플릿 디렉토리 경로
        
    Returns:
        Dict[str, str]: {파일명: HTML내용} 형태의 딕셔너리
    """
    templates = {}
    html_files = get_html_template_files(template_dir)
    
    for filename in html_files:
        content = get_html_template_content(template_dir, filename)
        if content:
            templates[filename] = content
    
    return templates


# 기본 템플릿 디렉토리
# 현재 스크립트의 디렉토리를 기준으로 절대 경로 생성
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "slide_template", "template1")
# DEFAULT_TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "slide_template", "html_slides_1")


def get_html_template_list() -> str:
    """사용 가능한 HTML 템플릿 파일들의 리스트를 조회합니다."""
    import json
    
    # info.json 파일 경로
    info_json_path = os.path.join(DEFAULT_TEMPLATE_DIR, "info.json")
    
    try:
        # info.json 파일 읽기
        with open(info_json_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        
        templates = info_data.get("templates", [])
        
        if not templates:
            return "사용 가능한 HTML 템플릿 파일이 없습니다."
        
        result = f"사용 가능한 HTML 템플릿 파일들 ({len(templates)}개):\n\n"
        
        for i, template in enumerate(templates, 1):
            filename = template.get("filename", "")
            description = template.get("description", "")
            role = template.get("role", "")
            layout = template.get("layout", "")
            use_case = template.get("use_case", "")
            
            result += f"{i}. {filename}\n"
            result += f"   역할: {role}\n"
            result += f"   레이아웃: {layout}\n"
            result += f"   사용 사례: {use_case}\n"
            result += f"   설명: {description}\n\n"
        
        return result
        
    except FileNotFoundError:
        # info.json이 없으면 기존 방식으로 fallback
        html_files = get_html_template_files(DEFAULT_TEMPLATE_DIR)
        
        if not html_files:
            return "사용 가능한 HTML 템플릿 파일이 없습니다."
        
        result = f"사용 가능한 HTML 템플릿 파일들 ({len(html_files)}개):\n"
        for i, filename in enumerate(html_files, 1):
            result += f"{i}. {filename}\n"
        
        return result
        
    except Exception as e:
        print(f"info.json 파일을 읽는 중 오류 발생: {e}")
        # 오류 발생 시 기존 방식으로 fallback
        html_files = get_html_template_files(DEFAULT_TEMPLATE_DIR)
        
        if not html_files:
            return "사용 가능한 HTML 템플릿 파일이 없습니다."
        
        result = f"사용 가능한 HTML 템플릿 파일들 ({len(html_files)}개):\n"
        for i, filename in enumerate(html_files, 1):
            result += f"{i}. {filename}\n"
        
        return result


@tool
def get_html_template_content_tool(filename: str) -> str:
    """지정된 HTML 템플릿 파일의 내용을 가져옵니다.
    
    Args:
        filename: 읽어올 HTML 템플릿 파일명 (확장자 포함)
    """
    content = get_html_template_content(DEFAULT_TEMPLATE_DIR, filename)
    
    if content is None:
        return f"파일 '{filename}'을 찾을 수 없거나 읽을 수 없습니다."
    
    return f"HTML 템플릿 '{filename}' 내용:\n\n{content}"


if __name__ == '__main__':
    result = get_html_template_list()
    print(result)

    html = get_html_template_content(DEFAULT_TEMPLATE_DIR, "index.html")
    print(html)