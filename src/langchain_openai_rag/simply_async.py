import asyncio


async def greet():
    print("hi")

async def main():
    g = greet()
    print(g)          # <coroutine object greet at 0x...> — nothing printed yet!
    await g

if __name__ == "__main__":
    asyncio.run(main())
