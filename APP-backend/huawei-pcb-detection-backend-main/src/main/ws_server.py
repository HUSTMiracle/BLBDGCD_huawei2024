import asyncio
import websockets
import threading
import sys
import os
import logging
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../util")

from util.iot import IotDeviceUtil


class WsServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        

    def run(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        asyncio.get_event_loop().run_until_complete(
            websockets.serve(self.server, "0.0.0.0", 5002)
        )
        asyncio.get_event_loop().run_forever()

    async def server(self, websocket, path):
        prev_data_text = ""
        while websocket.open:
            try:
                resp = IotDeviceUtil().get_device_shadow(path[1:])
                resp_data = resp.json()["shadow"][0]["reported"]["properties"]
                if prev_data_text != f"{resp_data}":
                    prev_data_text = f"{resp_data}"
                    marshalled_data = json.dumps(resp_data)
                    await websocket.send(marshalled_data)
                self.logger.error(f"ws_server: {resp_data}")
                await asyncio.sleep(1)
            except Exception as e:
                await websocket.send(f"Error: {str(e)}")
                await asyncio.sleep(1)
                continue