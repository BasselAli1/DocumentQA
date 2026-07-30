import asyncio
async def greet():
    print("hi")

g = greet()
print(g)          # <coroutine object greet at 0x...> — nothing printed yet!
await g) 