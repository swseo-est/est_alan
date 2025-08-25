import pytest
from pathlib import Path
from estalan.messages.format.chat_html import create_img_grid


def test_create_img_grid_basic():
    """기본적인 이미지 그리드 생성 테스트"""
    image_list = ["image.png", "image.png", "image.png", "image.png"]
    cols = 2
    
    result = create_img_grid(image_list, cols)
    
    # 결과가 HTML 문자열인지 확인
    assert isinstance(result, str)
    assert result.startswith('<div')
    assert result.endswith('</div>')
    
    # CSS 그리드 속성이 올바르게 설정되었는지 확인
    assert 'display: grid' in result
    assert 'grid-template-columns: repeat(2, 1fr)' in result
    
    # 이미지 태그가 올바른 개수만큼 생성되었는지 확인
    assert result.count('<img') == 4
    assert 'image.png' in result
    
    # result를 그대로 HTML 파일로 저장
    output_path = Path(__file__).parent / 'test_create_img_grid_basic.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"HTML 파일이 생성되었습니다: {output_path}")


def test_create_img_grid_edge_cases():
    """경계 케이스 테스트 (빈 리스트, 잘못된 입력 등)"""
    # 빈 이미지 리스트 테스트
    result_empty = create_img_grid([], 3)
    assert result_empty == '<div></div>'
    
    # 잘못된 열 수 테스트 (0 이하)
    result_invalid_cols = create_img_grid(["image.png"], 0)
    assert result_invalid_cols == '<div></div>'
    
    result_invalid_cols_negative = create_img_grid(["image.png"], -1)
    assert result_invalid_cols_negative == '<div></div>'
    
    # 이미지가 1개이고 열이 1개인 경우
    result_single = create_img_grid(["image.png"], 1)
    assert result_single.count('<img') == 1
    assert 'image.png' in result_single
    
    # result_single을 그대로 HTML 파일로 저장
    output_path = Path(__file__).parent / 'test_create_img_grid_edge_cases.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result_single)
    
    print(f"HTML 파일이 생성되었습니다: {output_path}")
