from pathlib import Path
import json
import logging
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import Response


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

logger = logging.getLogger("ai-service")


# ============================================================
# PATH
# ============================================================

# File này nằm tại:
# AI-models/service/main.py
#
# Parent:
# AI-models/service
#
# Parent.parent:
# AI-models

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
SCHEMA_PATH = MODEL_DIR / "schema.json"


# ============================================================
# LOAD MODEL + METADATA + SCHEMA
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy model tại: {MODEL_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy metadata tại: {METADATA_PATH}"
    )

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy schema tại: {SCHEMA_PATH}"
    )


model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Penguin AI Service",
    version="1.0.0"
)


# ============================================================
# REQUEST LOGGING + X-REQUEST-ID
# ============================================================

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next
):
    print(
        ">>> AI REQUEST RECEIVED:",
        request.method,
        request.url.path,
        flush=True
    )
    # --------------------------------------------------------
    # Lấy X-Request-ID từ request
    # --------------------------------------------------------

    request_id = request.headers.get("X-Request-ID")

    # Nếu request chưa có ID thì tự tạo
    if not request_id:
        request_id = str(uuid.uuid4())

    # Lưu lại để endpoint có thể sử dụng
    request.state.request_id = request_id

    # --------------------------------------------------------
    # Bắt đầu đo thời gian request
    # --------------------------------------------------------

    start_time = time.perf_counter()

    status_code = 500

    try:

        response = await call_next(request)

        status_code = response.status_code

    except Exception:

        status_code = 500

        raise

    finally:

        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        # ----------------------------------------------------
        # Structured JSON log
        # ----------------------------------------------------

        log_data = {
            "service": "ai-service",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": duration_ms
        }

        logging.getLogger("uvicorn.error").info(
            json.dumps(
                log_data,
                ensure_ascii=False
            )
        )

    # --------------------------------------------------------
    # Trả X-Request-ID về response
    # --------------------------------------------------------

    response.headers["X-Request-ID"] = request_id

    return response


# ============================================================
# STARTUP LOG
# ============================================================

logger.info(
    json.dumps(
        {
            "service": "ai-service",
            "event": "startup",
            "model_loaded": model is not None,
            "model_name": metadata.get("model_name"),
            "model_version": metadata.get("model_version")
        },
        ensure_ascii=False
    )
)


# ============================================================
# INPUT SCHEMA
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
def health():

    return {
        "status": "ok",
        "service": "ai-service",
        "model_loaded": model is not None,
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version")
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "trained_at": metadata.get("trained_at"),
        "labels": metadata.get("labels"),
        "metrics_test": metadata.get("metrics_test"),
        "library_versions": metadata.get("library_versions"),
        "schema": schema
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(payload: PenguinInput):

    # --------------------------------------------------------
    # Validate categorical values
    # --------------------------------------------------------

    allowed_islands = [
        "Biscoe",
        "Dream",
        "Torgersen"
    ]

    allowed_sex = [
        "MALE",
        "FEMALE"
    ]

    if payload.island not in allowed_islands:

        raise HTTPException(
            status_code=422,
            detail=f"island phải thuộc: {allowed_islands}"
        )

    if payload.sex not in allowed_sex:

        raise HTTPException(
            status_code=422,
            detail=f"sex phải thuộc: {allowed_sex}"
        )

    # --------------------------------------------------------
    # Convert request -> DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "culmen_length_mm": payload.culmen_length_mm,
        "culmen_depth_mm": payload.culmen_depth_mm,
        "flipper_length_mm": payload.flipper_length_mm,
        "body_mass_g": payload.body_mass_g,

        "island": payload.island,
        "sex": payload.sex

    }])

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        raw_prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi chạy model: {str(e)}"
        )

    # --------------------------------------------------------
    # Convert label
    # --------------------------------------------------------

    labels = metadata.get(
        "labels",
        [
            "Adelie",
            "Chinstrap",
            "Gentoo"
        ]
    )

    if isinstance(raw_prediction, int):

        prediction = labels[raw_prediction]

    else:

        prediction = str(raw_prediction)

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability_result = {}

    for index, probability in enumerate(
        probabilities
    ):

        if index < len(labels):

            label = labels[index]

            probability_result[label] = float(
                probability
            )

    confidence = float(
        max(probabilities)
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "prediction": prediction,

        "confidence": confidence,

        "probabilities": probability_result,

        "model_name": metadata.get(
            "model_name"
        ),

        "model_version": metadata.get(
            "model_version"
        )

    }
