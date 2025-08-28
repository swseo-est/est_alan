import asyncio
import logging
from starlette.applications import Starlette
from langgraph_runtime.lifespan import lifespan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

async def main():
    logger.info("[Migration] Starting migration via lifespan...")

    # 서버 코드에서 쓰는 것과 동일하게 lifespan을 붙인 Starlette 앱 생성
    app = Starlette(lifespan=lifespan)

    # 기존 서버에서 app.router.lifespan_context 로 실행하던 방식 그대로 사용
    async with app.router.lifespan_context(app):
        pass

    logger.info("[Migration] Migrations completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
