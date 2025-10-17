#!/usr/bin/env python3
import asyncio
import json
import os

import websockets

URI = os.environ.get("OPENPI_WS", "ws://127.0.0.1:8000")


def zeros_obs():
    H = W = 224

    def img():
        return [[[0] * 3 for _ in range(W)] for __ in range(H)]

    return {
        "images": {
            "base_0_rgb": img(),
            "left_wrist_0_rgb": img(),
            "right_wrist_0_rgb": img(),
        },
        "state": [0.0] * 14,
        "prompt": "pick up the red block",
    }


async def try_envelope(envelope):
    async with websockets.connect(URI, max_size=None) as ws:
        payload = {envelope: zeros_obs()}
        await ws.send(json.dumps(payload))
        raw = await ws.recv()
        try:
            msg = json.loads(raw)
        except Exception:
            print(f"[{envelope}] raw reply:", (raw[:200] + "...") if isinstance(raw, str) and len(raw) > 200 else raw)
            return True
        print(f"[{envelope}] reply keys:", list(msg.keys()))
        if "actions" in msg:
            act = msg["actions"]
            if isinstance(act, list):
                ah = len(act)
                ad = len(act[0]) if ah > 0 and isinstance(act[0], list) else "?"
                print(f"[{envelope}] actions shape ≈ ({ah}, {ad})")
            print(f"[{envelope}] ✓ inference path works")
        else:
            print(f"[{envelope}] got reply (no 'actions' key); server may wrap response differently.")
        return True


async def main():
    print("Connecting to", URI)
    for env in ("inputs", "obs"):
        try:
            ok = await try_envelope(env)
            if ok:
                return
        except Exception as e:
            print(f"[{env}] failed: {e!r}")
    print("✗ Both envelopes failed. Check the expected top-level key in src/openpi/serving/websocket_policy_server.py.")


if __name__ == "__main__":
    asyncio.run(main())
