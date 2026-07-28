import asyncio
import json
import websockets

async def main():
    async with websockets.connect(
        "ws://127.0.0.1:8766/stt/stream", open_timeout=120, close_timeout=5
    ) as ws:
        print("opened")
        meta = json.dumps({"sampleRate": 16000}).encode()
        for _ in range(5):
            await ws.send(len(meta).to_bytes(4, "little") + meta + (b"\x00\x00" * 512))
            await asyncio.sleep(0.05)
        print("still-open", ws.state.name)

asyncio.run(main())
print("ok")
