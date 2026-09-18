"""
api_client — REST / WebSocket client utilities.

Planned responsibilities
------------------------
* Provide a thin HTTP client for sending events and alerts to the backend API.
* Provide a WebSocket client for streaming detections / events in real time.
* Handle authentication tokens and retry logic.

Status: NOT YET IMPLEMENTED.

Sub-modules (planned)
---------------------
http_client.py      : Async HTTP client (httpx / aiohttp)
ws_client.py        : WebSocket streaming client
auth.py             : Token management
models.py           : Request / response DTOs

Dependencies (planned)
----------------------
* httpx or aiohttp  (async HTTP)
* websockets
* NO dependency on src.detection or src.tracking
  (this module consumes serialised data, not live objects)
"""

# NOT YET IMPLEMENTED
