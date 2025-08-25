def create_img_grid(image_list, cols):
    """
    이미지 리스트와 가로 열 수를 입력받아 이미지 그리드 HTML을 생성합니다.
    세로 행 수는 이미지 개수에 맞춰 자동으로 계산됩니다.
    
    Args:
        image_list (list): 이미지 URL 리스트
        cols (int): 가로 열 수
    
    Returns:
        str: 이미지 그리드 HTML 문자열
    """
    # 이미지 리스트가 비어있거나 cols가 0 이하인 경우 빈 div 반환
    if not image_list or cols <= 0:
        return '<div></div>'
    
    # 이미지 개수에 맞춰 필요한 행 수 계산
    rows = (len(image_list) + cols - 1) // cols  # 올림 나눗셈
    
    # 그리드 CSS 설정
    grid_style = f"display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 10px;"
    
    # 이미지 HTML 생성
    img_html = ""
    for img_url in image_list:
        img_html += f'<img src="{img_url}" style="width:100%; height:auto; object-fit:cover;"/>'
    
    msg = f"""<div style="{grid_style}">
        {img_html}
    </div>"""
    
    return msg
