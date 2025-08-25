from estalan.deployment.cli import main


if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # 기본 설정값
    default_port = "2020"
    default_config = "graph.json"
    
    # 환경변수에서 포트와 설정 파일 경로 가져오기
    port = os.environ.get("ESTALAN_PORT", default_port)
    config = os.environ.get("ESTALAN_CONFIG", default_config)
    
    # 명령행 인자가 있으면 우선 사용
    if len(sys.argv) > 1:
        # 명령행 인자를 파싱하여 포트와 설정 파일 경로 추출
        args = sys.argv[1:]
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                port = args[i + 1]
            elif arg == "--config" and i + 1 < len(args):
                config = args[i + 1]
    
    # CLI 인자로 전달
    sys.argv = [
        "estalan.deployment.cli",  # 스크립트 이름
        "--config", config,
        "--port", port
    ]

    main()
