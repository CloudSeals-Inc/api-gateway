"""
MIBA — API Gateway
Single entry point for the Collector and Citizen mobile apps.
Routes traffic to: classification-api · carbon-engine · token-engine · supervisor
"""

import os
import logging
import base64
import hashlib
from typing import Optional
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default to common dev origins if not set
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8000,https://miba-ui-460918627115.europe-west1.run.app").split(",")

app = FastAPI(title="MIBA API Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"]
)

# All routes are mounted under /api to match frontend's API_BASE_URL = http://localhost:8000/api
router = APIRouter(prefix="/api")


# Downstream service URLs
CLASSIFICATION_URL = os.getenv("CLASSIFICATION_API_URL", "http://classification-api:8080")
CARBON_URL         = os.getenv("CARBON_ENGINE_URL",      "http://carbon-engine:8080")
TOKEN_URL          = os.getenv("TOKEN_ENGINE_URL",        "http://token-engine:8080")
SUPERVISOR_URL     = os.getenv("SUPERVISOR_URL",          "http://supervisor:8080")

TIMEOUT = httpx.Timeout(60.0) # Longer timeout for AI


async def _request(method: str, url: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await getattr(client, method)(url, **kwargs)
        if resp.status_code >= 400:
            logger.error(f"Error from upstream {url} [{resp.status_code}]: {resp.text[:500]}")
        return resp

async def proxy(method: str, url: str, **kwargs):
    """Proxy and return as dict (for internal logic)."""
    resp = await _request(method, url, **kwargs)
    try:
        return resp.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON from {url}: {e} | Content={resp.text[:200]}")
        raise HTTPException(502, f"Invalid JSON from upstream {url}")

async def proxy_resp(method: str, url: str, **kwargs):
    """Proxy and return a full Response object (for final output)."""
    resp = await _request(method, url, **kwargs)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers={"Content-Type": resp.headers.get("Content-Type", "application/json")}
    )


# ─── Health ────────────────────────────────────────────────────────────────
@app.get("/health")  # keep at root so docker healthcheck still works
@router.get("/health")
async def health():
    statuses = {}
    for name, url in [("classification", CLASSIFICATION_URL), ("carbon", CARBON_URL),
                       ("tokens", TOKEN_URL), ("supervisor", SUPERVISOR_URL)]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as c:
                r = await c.get(f"{url}/health")
                statuses[name] = "ok" if r.status_code == 200 else "degraded"
        except Exception:
            statuses[name] = "unreachable"
    overall = "ok" if all(v == "ok" for v in statuses.values()) else "degraded"
    return {"status": overall, "services": statuses}


# ─── Internal Helpers ──────────────────────────────────────────────────────
def decode_base64_image(b64_string: str) -> bytes:
    """Robustly decode base64 image strings from browsers."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    
    # Strip whitespace/newlines and fix padding
    b64_string = b64_string.strip().replace(" ", "+").replace("\n", "").replace("\r", "")
    missing_padding = len(b64_string) % 4
    if missing_padding:
        b64_string += "=" * (4 - missing_padding)
        
    return base64.b64decode(b64_string)


# ─── AI Analyze (Live Detection) ──────────────────────────────────────────
@router.post("/ai/analyze")
async def ai_analyze(payload: dict):
    """UI uses this for real-time analysis before submitting report."""
    img_b64 = payload.get("imageUrl")
    if not img_b64:
        raise HTTPException(400, "imageUrl required")
    
    try:
        logger.info(f"AI Analyze: decoding payload (len={len(img_b64)})")
        raw = decode_base64_image(img_b64)
        logger.info(f"AI Analyze: decoded bytes (len={len(raw)})")
    except Exception as e:
        logger.error(f"AI Analyze: base64 decoding failed: {e} | start={img_b64[:50]}")
        raise HTTPException(400, "Invalid base64 image data")

    return await proxy("post", f"{CLASSIFICATION_URL}/classify",
        files={"image": ("upload.jpg", raw, "image/jpeg")},
    )


# ─── Reports (Citizen) ────────────────────────────────────────────────────
@router.get("/reports")
async def get_reports():
    """UI dashboard list."""
    resp = await proxy("get", f"{SUPERVISOR_URL}/workorder")
    return resp.get("work_orders", [])

@router.post("/reports")
async def create_report(payload: dict):
    """UI report submission."""
    img_b64 = payload.get("imageUrl")
    raw = decode_base64_image(img_b64) if img_b64 else None
    cls_result = await proxy("post", f"{CLASSIFICATION_URL}/classify",
        files={"image": ("report.jpg", raw, "image/jpeg")} if raw else None,
    )
    return await proxy("post", f"{SUPERVISOR_URL}/workorder/create", json={
        "reporter_id": payload.get("reporterPhone", "anonymous"),
        "report_lat": 0.0,
        "report_lng": 0.0,
        "report_photo_hash": cls_result.get("image_hash", ""),
        "classification_result": cls_result,
    })


# ─── AI Analytics & Chat ──────────────────────────────────────────────────
@router.get("/ai/analytics")
async def get_analytics():
    return await proxy("get", f"{SUPERVISOR_URL}/ai/analytics")

@router.post("/ai/analytics/chat")
async def ai_chat(payload: dict):
    return await proxy("post", f"{SUPERVISOR_URL}/ai/analytics/chat", json=payload)


# ─── Legacy / Mobile App Routes ───────────────────────────────────────────
@router.post("/report")
async def report_garbage(
    image: UploadFile = File(...),
    reporter_id: str = "anonymous",
    lat: float = 0.0,
    lng: float = 0.0,
):
    raw = await image.read()
    cls_result = await proxy("post", f"{CLASSIFICATION_URL}/classify",
        files={"image": (image.filename, raw, image.content_type)},
        data={"location_lat": str(lat), "location_lng": str(lng)},
    )
    wo_result = await proxy("post", f"{SUPERVISOR_URL}/workorder/create", json={
        "reporter_id": reporter_id, "report_lat": lat, "report_lng": lng,
        "report_photo_hash": cls_result.get("image_hash", ""),
        "classification_result": cls_result,
    })
    return {
        "work_order_id": wo_result.get("work_order_id"),
        "status": wo_result.get("status"),
        "classification_summary": {
            "dominant_category": cls_result.get("dominant_category"),
            "total_weight_kg": cls_result.get("total_weight_kg_estimate"),
            "categories_found": cls_result.get("categories_found"),
            "recyclable_fraction": cls_result.get("recyclable_fraction"),
        },
        "message": "Garbage reported successfully.",
    }

@router.post("/pickup/complete")
async def complete_pickup(payload: dict):
    return await proxy("post", f"{SUPERVISOR_URL}/workorder/verify", json=payload)

@router.get("/collector/{collector_id}/wallet")
async def get_wallet(collector_id: str):
    return await proxy("get", f"{TOKEN_URL}/tokens/ledger/{collector_id}")

@router.get("/workorder/{work_order_id}")
async def get_work_order(work_order_id: str):
    return await proxy("get", f"{SUPERVISOR_URL}/workorder/{work_order_id}")

@router.get("/carbon/table")
async def carbon_table():
    return await proxy("get", f"{CARBON_URL}/carbon/table")

@router.get("/categories")
async def categories():
    return await proxy("get", f"{CLASSIFICATION_URL}/categories")


# ─── Auth (Registration & Login) ─────────────────────────────────────────
@router.post("/auth/register")
async def auth_register(payload: dict):
    return await proxy_resp("post", f"{TOKEN_URL}/auth/register", json=payload)

@router.post("/auth/login")
async def auth_login(payload: dict):
    return await proxy_resp("post", f"{TOKEN_URL}/auth/login", json=payload)

app.include_router(router)
