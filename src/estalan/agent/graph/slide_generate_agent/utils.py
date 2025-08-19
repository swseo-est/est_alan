import os
import asyncio
import threading
import json
from typing import List, Dict, Optional
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.tools import tool


# 스레드 안전한 캐시를 위한 락
_template_cache_lock = threading.Lock()
_template_cache: Dict[str, str] = {}


def clear_template_cache():
    """템플릿 캐시를 초기화합니다."""
    with _template_cache_lock:
        _template_cache.clear()


# 기본 템플릿 디렉토리
# 현재 스크립트의 디렉토리를 기준으로 절대 경로 생성
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_template_dir(template_folder: str = "general") -> str:
    """
    템플릿 폴더명을 받아서 전체 경로를 반환합니다.
    
    Args:
        template_folder (str): 템플릿 폴더명 (예: "general", "compare")
        
    Returns:
        str: 템플릿 디렉토리의 전체 경로
    """
    return os.path.join(SCRIPT_DIR, "slide_template", template_folder)


def get_all_template_folders() -> List[str]:
    """
    slide_template 디렉토리 내의 info.json 파일이 있는 템플릿 폴더명만 반환합니다.
    
    Returns:
        List[str]: info.json 파일이 있는 템플릿 폴더명 리스트
    """
    template_base_dir = os.path.join(SCRIPT_DIR, "slide_template")
    try:
        folders = []
        for item in os.listdir(template_base_dir):
            item_path = os.path.join(template_base_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # info.json 파일이 있는지 확인
                info_json_path = os.path.join(item_path, "info.json")
                if os.path.exists(info_json_path):
                    folders.append(item)
        return sorted(folders)
    except Exception as e:
        print(f"템플릿 폴더 목록을 읽는 중 오류 발생: {e}")
        return []


def extract_template_metadata_from_info_json(template_folder: str) -> Optional[Dict]:
    """
    지정된 템플릿 폴더의 info.json 파일을 읽어서 template_library.metadata 정보를 추출합니다.
    
    Args:
        template_folder (str): 템플릿 폴더명
        
    Returns:
        Optional[Dict]: metadata 정보 또는 None (파일이 없거나 읽기 실패 시)
    """
    try:
        info_json_path = os.path.join(get_template_dir(template_folder), "info.json")
        
        if not os.path.exists(info_json_path):
            print(f"info.json 파일이 존재하지 않습니다: {info_json_path}")
            return None
        
        with open(info_json_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        
        # template_library.metadata 추출
        if "template_library" in info_data:
            metadata = info_data["template_library"].get("metadata", {})
            return metadata
        else:
            print(f"template_library 정보를 찾을 수 없습니다: {template_folder}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류 ({template_folder}): {e}")
        return None
    except Exception as e:
        print(f"info.json 읽기 실패 ({template_folder}): {e}")
        return None


def format_template_info(template_folder: str) -> str:
    """
    지정된 템플릿 폴더의 info.json을 읽어서 지정된 형식의 문자열을 반환합니다.
    
    Args:
        template_folder (str): 템플릿 폴더명
        
    Returns:
        str: 포맷된 정보 문자열
    """
    try:
        info_json_path = os.path.join(get_template_dir(template_folder), "info.json")
        
        if not os.path.exists(info_json_path):
            return f"info.json 파일이 존재하지 않습니다: {template_folder}"
        
        with open(info_json_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        
        # template_library 정보 추출
        if "template_library" not in info_data:
            return f"template_library 정보를 찾을 수 없습니다: {template_folder}"
        
        template_library = info_data["template_library"]
        metadata = template_library.get("metadata", {})
        templates = template_library.get("templates", [])
        
        # 결과 문자열 생성
        result = f"template_dir: {template_folder}\n"
        
        # themes
        themes = metadata.get("themes", [])
        result += f"themes: {', '.join(themes) if themes else 'N/A'}\n"
        
        # styles
        styles = metadata.get("styles", [])
        result += f"styles: {', '.join(styles) if styles else 'N/A'}\n"
        
        # included_template
        result += "included_template\n"
        
        # 템플릿 목록
        if templates:
            for template in templates:
                template_name = template.get("name", "Unknown")
                template_key = template.get("key", "unknown")
                result += f"- {template_name} ({template_key})\n"
        else:
            result += "- No templates found\n"
        
        return result
        
    except json.JSONDecodeError as e:
        return f"JSON 파싱 오류 ({template_folder}): {e}"
    except Exception as e:
        return f"info.json 읽기 실패 ({template_folder}): {e}"


def get_all_templates_info() -> str:
    """
    모든 템플릿 폴더의 정보를 format_template_info를 이용해서 읽어서 반환합니다.
    
    Returns:
        str: 모든 템플릿의 포맷된 정보 문자열
    """
    template_folders = get_all_template_folders()
    
    if not template_folders:
        return "info.json 파일이 있는 템플릿 폴더를 찾을 수 없습니다."
    
    result = f"📁 전체 템플릿 정보 ({len(template_folders)}개 폴더)\n"
    result += "=" * 80 + "\n\n"
    
    for i, folder in enumerate(template_folders, 1):
        result += f"🔸 템플릿 {i}/{len(template_folders)}\n"
        result += format_template_info(folder)
        result += "\n" + "-" * 60 + "\n\n"
    
    return result


def get_template_metadata_string(template_folder: str) -> str:
    """
    지정된 템플릿 폴더의 info.json을 읽고 등록된 template의 메타데이터를 한번에 string으로 출력합니다.
    
    Args:
        template_folder (str): 템플릿 폴더명
        
    Returns:
        str: 템플릿 메타데이터 문자열
    """
    try:
        info_json_path = os.path.join(get_template_dir(template_folder), "info.json")
        
        if not os.path.exists(info_json_path):
            return f"info.json 파일이 존재하지 않습니다: {template_folder}"
        
        with open(info_json_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        
        # template_library 정보 추출
        if "template_library" not in info_data:
            return f"template_library 정보를 찾을 수 없습니다: {template_folder}"
        
        template_library = info_data["template_library"]
        metadata = template_library.get("metadata", {})
        templates = template_library.get("templates", [])
        
        # 결과 문자열 생성
        result = f"📂 {template_folder} - 템플릿 메타데이터\n"
        result += "=" * 60 + "\n\n"
        
        # 기본 메타데이터 정보
        result += "📋 기본 정보:\n"
        result += f"   - 버전: {metadata.get('version', 'N/A')}\n"
        result += f"   - 생성일: {metadata.get('created', 'N/A')}\n"
        result += f"   - 총 템플릿 수: {metadata.get('total_templates', 'N/A')}\n"
        result += f"   - 소스명: {metadata.get('source_name', 'N/A')}\n"
        result += f"   - 해상도: {metadata.get('resolution', {}).get('width', 'N/A')}x{metadata.get('resolution', {}).get('height', 'N/A')}\n"
        result += f"   - 테마: {', '.join(metadata.get('themes', []))}\n"
        result += f"   - 스타일: {', '.join(metadata.get('styles', []))}\n\n"
        
        # 등록된 템플릿 목록
        result += f"🎨 등록된 템플릿 ({len(templates)}개):\n"
        result += "-" * 40 + "\n"
        
        if templates:
            for i, template in enumerate(templates, 1):
                template_id = template.get("id", "N/A")
                template_key = template.get("key", "unknown")
                template_name = template.get("name", "Unknown")
                template_name_korean = template.get("name_korean", "")
                template_category = template.get("category", "N/A")
                
                result += f"{i:2d}. [{template_id}] {template_name}"
                if template_name_korean:
                    result += f" ({template_name_korean})"
                result += f"\n"
                result += f"     파일이름: {template_key}.html\n"
                result += f"     카테고리: {template_category}\n"
                
                # 레이아웃 정보
                layout = template.get("layout", {})
                if layout:
                    layout_type = layout.get("type", "N/A")
                    result += f"     레이아웃: {layout_type}\n"
                
                # 사용 사례
                use_cases = template.get("use_cases", [])
                if use_cases:
                    result += f"     사용 사례: {', '.join(use_cases[:3])}"  # 처음 3개만 표시
                    if len(use_cases) > 3:
                        result += f" ... (+{len(use_cases)-3}개 더)"
                    result += "\n"
                
                result += "\n"
        else:
            result += "   등록된 템플릿이 없습니다.\n\n"
        
        return result
        
    except json.JSONDecodeError as e:
        return f"JSON 파싱 오류 ({template_folder}): {e}"
    except Exception as e:
        return f"info.json 읽기 실패 ({template_folder}): {e}"


@tool
def get_html_template_content_tool(filename: str, template_folder: str = "general") -> str:
    """지정된 HTML 템플릿 파일의 내용을 가져옵니다.
    
    Args:
        filename: 읽어올 HTML 템플릿 파일명 (확장자 포함)
        template_folder: 템플릿 폴더명 (기본값: "general")
    """
    template_dir = get_template_dir(template_folder)
    content = get_html_template_content(template_dir, filename)
    
    if content is None:
        msg = f"파일 '{filename}'을 찾을 수 없거나 읽을 수 없습니다."
        print(msg)
        return msg
    
    return f"{content}"


if __name__ == "__main__":
    print(get_all_template_folders())

    print(extract_template_metadata_from_info_json("1.현대 AI 기술"))

    print(format_template_info("1.현대 AI 기술"))

    print(get_all_templates_info())

    print(get_template_metadata_string("1.현대 AI 기술"))