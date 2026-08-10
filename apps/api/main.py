"""HTTP API for training a Self-Organising Map from a CSV upload."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jwt import PyJWKClient
from minio import Minio
from minio.error import S3Error

from som_core import train_som_from_csv

# Prefer WEB_ROOT in containers; fall back to monorepo apps/web next to apps/api
_DEFAULT_WEB = Path(__file__).resolve().parents[1] / "web"
UI_DIR = Path(os.getenv("WEB_ROOT", str(_DEFAULT_WEB)))

logger = logging.getLogger(__name__)

ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "/app/artifacts"))
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "som-artifacts")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", "http://localhost:9010").rstrip("/")

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8180").rstrip("/")
KEYCLOAK_INTERNAL_URL = os.getenv(
    "KEYCLOAK_INTERNAL_URL", KEYCLOAK_URL
).rstrip("/")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "som")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "som-ui")
KEYCLOAK_ISSUER = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}"
KEYCLOAK_JWKS_URL = (
    f"{KEYCLOAK_INTERNAL_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
)

# Concurrency controls (per API process / uvicorn worker)
MAX_CONCURRENT_TRAINS = max(1, int(os.getenv("MAX_CONCURRENT_TRAINS", "2")))
TRAIN_QUEUE_TIMEOUT_SEC = float(os.getenv("TRAIN_QUEUE_TIMEOUT_SEC", "30"))
TRAIN_THREAD_WORKERS = max(
    1, int(os.getenv("TRAIN_THREAD_WORKERS", str(MAX_CONCURRENT_TRAINS)))
)

minio_client: Minio | None = None
jwks_client: PyJWKClient | None = None
train_semaphore: asyncio.Semaphore | None = None
train_executor: ThreadPoolExecutor | None = None
bearer_scheme = HTTPBearer(auto_error=True)


def get_minio() -> Minio:
    if minio_client is None:
        raise RuntimeError("MinIO client is not initialized")
    return minio_client


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("Created MinIO bucket %s", bucket)
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket}/*"],
            }
        ],
    }
    client.set_bucket_policy(bucket, json.dumps(policy))


def upload_artifact(local_path: Path, object_key: str) -> dict:
    client = get_minio()
    client.fput_object(
        MINIO_BUCKET,
        object_key,
        str(local_path),
        content_type="image/png",
    )
    path = f"s3://{MINIO_BUCKET}/{object_key}"
    url = f"{MINIO_PUBLIC_URL}/{MINIO_BUCKET}/{object_key}"
    return {"path": path, "object_key": object_key, "url": url}


def _run_training_job(
    *,
    csv_path: Path,
    job_id: str,
    job_dir: Path,
    columns: list[str],
    label_column: str | None,
    width: int,
    height: int,
    n_iterations: int,
    seed: int | None,
    scale: bool,
    online: bool,
    alpha0: float,
    requested_by: str | None,
) -> dict[str, Any]:
    """Blocking train + MinIO upload (runs in a worker thread)."""
    result = train_som_from_csv(
        csv_path,
        feature_columns=columns,
        label_column=label_column,
        width=width,
        height=height,
        n_iterations=n_iterations,
        seed=seed,
        scale=scale,
        online=online,
        alpha0=alpha0,
        output_dir=job_dir,
    )

    components_local = Path(result["artifacts"]["components"])
    bmu_local = Path(result["artifacts"]["bmu"])
    artifacts = {
        "components": upload_artifact(
            components_local, f"{job_id}/som_components.png"
        ),
        "bmu": upload_artifact(bmu_local, f"{job_id}/som_bmu.png"),
    }
    return {
        "job_id": job_id,
        "requested_by": requested_by,
        "n_samples": result["n_samples"],
        "n_features": result["n_features"],
        "feature_columns": result["feature_columns"],
        "label_column": result["label_column"],
        "map_size": result["map_size"],
        "n_iterations": result["n_iterations"],
        "online": result["online"],
        "quantization_error": result["quantization_error"],
        "weights_shape": result["weights_shape"],
        "artifacts": artifacts,
    }


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
) -> dict[str, Any]:
    if jwks_client is None:
        raise HTTPException(status_code=503, detail="Auth is not ready")
    token = credentials.credentials
    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=KEYCLOAK_ISSUER,
            options={"verify_aud": False},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    azp = payload.get("azp")
    aud = payload.get("aud")
    audiences = aud if isinstance(aud, list) else [aud] if aud else []
    if azp != KEYCLOAK_CLIENT_ID and KEYCLOAK_CLIENT_ID not in audiences:
        raise HTTPException(status_code=401, detail="Token was not issued for this client")
    return payload


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global minio_client, jwks_client, train_semaphore, train_executor
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    train_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRAINS)
    train_executor = ThreadPoolExecutor(
        max_workers=TRAIN_THREAD_WORKERS,
        thread_name_prefix="som-train",
    )
    logger.info(
        "Concurrency: max_trains=%s queue_timeout=%ss thread_workers=%s",
        MAX_CONCURRENT_TRAINS,
        TRAIN_QUEUE_TIMEOUT_SEC,
        TRAIN_THREAD_WORKERS,
    )

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )

    last_error: Exception | None = None
    for _ in range(30):
        try:
            ensure_bucket(minio_client, MINIO_BUCKET)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1)
    if last_error is not None:
        raise RuntimeError(f"Could not connect to MinIO: {last_error}") from last_error
    logger.info("MinIO ready at %s bucket=%s", MINIO_ENDPOINT, MINIO_BUCKET)

    jwks_client = PyJWKClient(KEYCLOAK_JWKS_URL, cache_keys=True)
    for _ in range(60):
        try:
            jwks_client.fetch_data()
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)
    if last_error is not None:
        logger.warning(
            "Keycloak JWKS not ready at startup (%s); will retry on first request",
            last_error,
        )
    else:
        logger.info("Keycloak JWKS ready issuer=%s", KEYCLOAK_ISSUER)

    yield

    if train_executor is not None:
        train_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Kohonen SOM API",
    description="Upload a CSV and train a Self-Organising Map. Images stored in MinIO.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
def ui():
    index = UI_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index)


@app.get("/config")
def app_config():
    """Public OIDC settings for the dashboard (no secrets)."""
    return {
        "keycloak": {
            "url": KEYCLOAK_URL,
            "realm": KEYCLOAK_REALM,
            "clientId": KEYCLOAK_CLIENT_ID,
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "minio_bucket": MINIO_BUCKET,
        "keycloak_issuer": KEYCLOAK_ISSUER,
        "max_concurrent_trains": MAX_CONCURRENT_TRAINS,
    }


@app.get("/me")
def me(user: Annotated[dict[str, Any], Depends(verify_token)]):
    return {
        "sub": user.get("sub"),
        "preferred_username": user.get("preferred_username"),
        "email": user.get("email"),
        "name": user.get("name") or user.get("preferred_username"),
    }


@app.post("/som/train")
async def som_train(
    user: Annotated[dict[str, Any], Depends(verify_token)],
    file: UploadFile = File(..., description="CSV file with a header row"),
    feature_columns: str = Form(
        ...,
        description='JSON array or comma-separated feature column names, e.g. ["a","b"] or a,b',
    ),
    label_column: str | None = Form(None),
    width: int = Form(10),
    height: int = Form(10),
    n_iterations: int = Form(1000),
    seed: int | None = Form(42),
    scale: bool = Form(True),
    online: bool = Form(False),
    alpha0: float = Form(0.1),
):
    if train_semaphore is None or train_executor is None:
        raise HTTPException(status_code=503, detail="Training pool is not ready")

    columns = _parse_feature_columns(feature_columns)
    if not columns:
        raise HTTPException(status_code=400, detail="feature_columns must not be empty")

    job_id = uuid.uuid4().hex[:12]
    job_dir = ARTIFACT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "data.csv").suffix or ".csv"
    csv_path = job_dir / f"input{suffix}"

    try:
        with csv_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Failed to store upload: {exc}"
        ) from exc

    try:
        await asyncio.wait_for(
            train_semaphore.acquire(),
            timeout=TRAIN_QUEUE_TIMEOUT_SEC,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Too many concurrent training jobs "
                f"(limit {MAX_CONCURRENT_TRAINS} per worker). Try again shortly."
            ),
        ) from exc

    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(
            train_executor,
            lambda: _run_training_job(
                csv_path=csv_path,
                job_id=job_id,
                job_dir=job_dir,
                columns=columns,
                label_column=label_column or None,
                width=width,
                height=height,
                n_iterations=n_iterations,
                seed=seed,
                scale=scale,
                online=online,
                alpha0=alpha0,
                requested_by=user.get("preferred_username") or user.get("sub"),
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except S3Error as exc:
        raise HTTPException(
            status_code=502, detail=f"MinIO upload failed: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        train_semaphore.release()

    return payload


def _parse_feature_columns(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON feature_columns: {exc}"
            ) from exc
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise HTTPException(
                status_code=400,
                detail="feature_columns JSON must be an array of strings",
            )
        return parsed
    return [c.strip() for c in raw.split(",") if c.strip()]
