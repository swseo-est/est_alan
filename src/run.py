from estalan.deployment.cli import main


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    sys.argv = [
        "estalan.deployment.cli",  # 스크립트 이름
        "--config", "graph.json",
    ]

    main()
