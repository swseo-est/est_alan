import os
import asyncio
import threading
from typing import List, Dict, Optional
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.tools import tool


# 스레드 안전한 캐시를 위한 락
_template_cache_lock = threading.Lock()
_template_cache: Dict[str, str] = {}


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
        
        # 스레드 안전한 파일 리스트 읽기
        with template_path.iterdir() as it:
            html_files = [f.name for f in it if f.is_file() and f.suffix.lower() == '.html']
        return sorted(html_files)
    
    except Exception as e:
        print(f"템플릿 파일 리스트를 읽는 중 오류 발생: {e}")
        return []


def get_html_template_content(template_dir: str, filename: str) -> Optional[str]:
    """
    지정된 템플릿 디렉토리에서 특정 HTML 파일의 내용을 읽어옵니다.
    캐싱을 통해 중복 파일 읽기를 방지합니다.
    
    Args:
        template_dir (str): 템플릿 디렉토리 경로
        filename (str): 읽어올 HTML 파일명
        
    Returns:
        Optional[str]: HTML 파일 내용 또는 None (파일이 없거나 읽기 실패 시)
    """
    cache_key = f"{template_dir}:{filename}"
    
    # 캐시에서 먼저 확인
    with _template_cache_lock:
        if cache_key in _template_cache:
            return _template_cache[cache_key]
    
    try:
        file_path = Path(template_dir) / filename
        if not file_path.exists():
            print(f"템플릿 파일이 존재하지 않습니다: {file_path}")
            return None
        
        # 파일 읽기 시 스레드 안전성 보장
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 캐시에 저장
        with _template_cache_lock:
            _template_cache[cache_key] = content
        
        return content
    
    except Exception as e:
        print(f"템플릿 파일을 읽는 중 오류 발생: {e}")
        return None


async def get_html_template_content_async(template_dir: str, filename: str) -> Optional[str]:
    """
    비동기적으로 HTML 템플릿 파일 내용을 읽어옵니다.
    
    Args:
        template_dir (str): 템플릿 디렉토리 경로
        filename (str): 읽어올 HTML 파일명
        
    Returns:
        Optional[str]: HTML 파일 내용 또는 None
    """
    loop = asyncio.get_event_loop()
    
    # ThreadPoolExecutor를 사용하여 파일 I/O를 별도 스레드에서 실행
    with ThreadPoolExecutor(max_workers=4) as executor:
        future = loop.run_in_executor(
            executor, 
            get_html_template_content, 
            template_dir, 
            filename
        )
        return await future


def get_all_html_templates(template_dir: str) -> Dict[str, str]:
    """
    지정된 템플릿 디렉토리의 모든 HTML 파일을 읽어서 파일명과 내용을 딕셔너리로 반환합니다.
    병렬처리를 통해 성능을 향상시킵니다.
    
    Args:
        template_dir (str): 템플릿 디렉토리 경로
        
    Returns:
        Dict[str, str]: {파일명: HTML내용} 형태의 딕셔너리
    """
    html_files = get_html_template_files(template_dir)
    templates = {}
    
    if not html_files:
        return templates
    
    # 병렬 처리를 위한 ThreadPoolExecutor 사용
    with ThreadPoolExecutor(max_workers=min(len(html_files), 4)) as executor:
        # 각 파일에 대해 비동기 작업 생성
        future_to_filename = {
            executor.submit(get_html_template_content, template_dir, filename): filename
            for filename in html_files
        }
        
        # 완료된 작업들을 수집
        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                content = future.result()
                if content:
                    templates[filename] = content
            except Exception as e:
                print(f"파일 '{filename}' 처리 중 오류 발생: {e}")
    
    return templates


async def get_all_html_templates_async(template_dir: str) -> Dict[str, str]:
    """
    비동기적으로 모든 HTML 템플릿을 읽어옵니다.
    
    Args:
        template_dir (str): 템플릿 디렉토리 경로
        
    Returns:
        Dict[str, str]: {파일명: HTML내용} 형태의 딕셔너리
    """
    html_files = get_html_template_files(template_dir)
    
    if not html_files:
        return {}
    
    # 비동기 작업들을 생성
    tasks = [
        get_html_template_content_async(template_dir, filename)
        for filename in html_files
    ]
    
    # 모든 작업을 병렬로 실행
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과를 딕셔너리로 변환
    templates = {}
    for filename, result in zip(html_files, results):
        if isinstance(result, Exception):
            print(f"파일 '{filename}' 처리 중 오류 발생: {result}")
        elif result is not None:
            templates[filename] = result
    
    return templates


def clear_template_cache():
    """템플릿 캐시를 초기화합니다."""
    with _template_cache_lock:
        _template_cache.clear()


# 기본 템플릿 디렉토리
# 현재 스크립트의 디렉토리를 기준으로 절대 경로 생성
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "slide_template", "template1")
DEFAULT_TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "slide_template", "morden template")


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