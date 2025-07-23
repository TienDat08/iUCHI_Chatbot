import asyncio
import uvicorn

async def main():
    config = uvicorn.Config("agents.app:app",
                            reload=True,
                            log_level="info"
                            )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())