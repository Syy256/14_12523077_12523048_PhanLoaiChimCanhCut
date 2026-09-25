import os
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient


# ============================================================
# CONFIG
# ============================================================

AI_SERVICE_URL = os.getenv(
    "AI_SERVICE_URL",
    "http://localhost:8001"
)

MONGO_URL = os.getenv(
    "MONGO_URL",
    "mongodb://localhost:27017"
)

MONGO_DB = os.getenv(
    "MONGO_DB",
    "penguin_app"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Penguin Backend API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MONGODB
# ============================================================

mongo_client = MongoClient(
    MONGO_URL,
    serverSelectionTimeoutMS=3000
)

db = mongo_client[MONGO_DB]

prediction_collection = db["predictions"]


# ============================================================
# INPUT MODEL
# ============================================================

class PenguinInput(BaseModel):

    culmen_length_mm: float = Field(gt=0)
    culmen_depth_mm: float = Field(gt=0)
    flipper_length_mm: float = Field(gt=0)
    body_mass_g: float = Field(gt=0)

    island: str = Field(min_length=1)
    sex: str = Field(min_length=1)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    ai_status = "unknown"
    mongo_status = "unknown"

    # Check AI Service
    try:

        async with httpx.AsyncClient(timeout=3.0) as client:

            response = await client.get(
                f"{AI_SERVICE_URL}/health"
            )

            if response.status_code == 200:
                ai_status = "ok"
            else:
                ai_status = "error"

    except Exception:
        ai_status = "error"

    # Check MongoDB
    try:

        mongo_client.admin.command("ping")
        mongo_status = "ok"

    except Exception:
        mongo_status = "error"

    return {
        "status": "ok",
        "service": "backend",
        "ai_service": ai_status,
        "mongodb": mongo_status
    }


# Alias để dễ test
@app.get("/api/health")
async def api_health():

    return await health()


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/api/model-info")
async def model_info():

    try:

        async with httpx.AsyncClient(timeout=5.0) as client:

            response = await client.get(
                f"{AI_SERVICE_URL}/model-info"
            )

        if response.status_code != 200:

            raise HTTPException(
                status_code=502,
                detail="AI Service không trả về model-info"
            )

        return response.json()

    except httpx.RequestError as e:

        raise HTTPException(
            status_code=503,
            detail=f"Không kết nối được AI Service: {str(e)}"
        )


# ============================================================
# PREDICT
# ============================================================

@app.post("/api/predict")
async def predict(payload: PenguinInput):

    data = payload.model_dump()

    # --------------------------------------------------------
    # Call AI Service
    # --------------------------------------------------------

    try:

        async with httpx.AsyncClient(timeout=10.0) as client:

            response = await client.post(
                f"{AI_SERVICE_URL}/predict",
                json=data
            )

    except httpx.RequestError as e:

        raise HTTPException(
            status_code=503,
            detail=f"Không kết nối được AI Service: {str(e)}"
        )

    # --------------------------------------------------------
    # AI error
    # --------------------------------------------------------

    if response.status_code != 200:

        try:
            detail = response.json()
        except Exception:
            detail = response.text

        raise HTTPException(
            status_code=response.status_code,
            detail=detail
        )

    result = response.json()

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_document = {
        "input": data,
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "probabilities": result.get("probabilities"),
        "model_name": result.get("model_name"),
        "model_version": result.get("model_version"),
        "created_at": datetime.now(timezone.utc)
    }

    history_saved = False

    try:

        prediction_collection.insert_one(
            history_document
        )

        history_saved = True

    except Exception:
        # Prediction vẫn có thể trả về nếu MongoDB
        # tạm thời không kết nối được.
        history_saved = False

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        **result,
        "created_at": history_document["created_at"].isoformat(),
        "history_saved": history_saved
    }


# ============================================================
# HISTORY
# ============================================================

@app.get("/api/history")
async def history():

    try:

        documents = prediction_collection.find(
            {},
            {
                "_id": 0
            }
        ).sort(
            "created_at",
            -1
        ).limit(50)

        items = list(documents)

        for item in items:

            if isinstance(
                item.get("created_at"),
                datetime
            ):
                item["created_at"] = (
                    item["created_at"]
                    .astimezone(timezone.utc)
                    .isoformat()
                )

        return {
            "items": items,
            "count": len(items)
        }

    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=f"Không đọc được history: {str(e)}"
        )