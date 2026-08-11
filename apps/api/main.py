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
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jwt import PyJWKClient
from minio import Minio
from minio.error import S3Error

from som_core import SOM, load_numeric_csv, train_som_from_csv

# Prefer WEB_ROOT in containers; fall back to monorepo apps/web next to apps/api
_DEFAULT_WEB = Path(__file__).resolve().parents[1] / "web"
UI_DIR = Path(os.getenv("WEB_ROOT", str(_DEFAULT_WEB)))

logger = logging.getLogger(__name__)

ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "/app/artifacts"))
MODEL_CACHE_ROOT = Path(os.getenv("MODEL_CACHE_ROOT", str(ARTIFACT_ROOT / "models")))
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


def upload_artifact(
    local_path: Path,
    object_key: str,
    content_type: str = "application/octet-stream",
) -> dict:
    client = get_minio()
    client.fput_object(
        MINIO_BUCKET,
        object_key,
        str(local_path),
        content_type=content_type,
    )
    path = f"s3://{MINIO_BUCKET}/{object_key}"
    url = f"{MINIO_PUBLIC_URL}/{MINIO_BUCKET}/{object_key}"
    return {"path": path, "object_key": object_key, "url": url}


def _model_object_keys(model_id: str) -> dict[str, str]:
    prefix = f"models/{model_id}"
    return {
        "model": f"{prefix}/model.npz",
        "meta": f"{prefix}/meta.json",
    }


def _write_meta_json(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _download_model_artifacts(model_id: str) -> tuple[Path, dict[str, Any]]:
    """Download model.npz + meta.json from MinIO into a local cache dir."""
    keys = _model_object_keys(model_id)
    cache_dir = MODEL_CACHE_ROOT / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "model.npz"
    meta_path = cache_dir / "meta.json"
    client = get_minio()
    try:
        client.fget_object(MINIO_BUCKET, keys["model"], str(model_path))
        client.fget_object(MINIO_BUCKET, keys["meta"], str(meta_path))
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchBucket"}:
            raise FileNotFoundError(f"Model {model_id!r} not found") from exc
        raise
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model_path, meta


def _load_inference_matrix(
    *,
    csv_path: Path | None,
    rows_json: str | None,
    feature_columns: list[str] | None,
    meta: dict[str, Any],
) -> tuple[Any, list[str]]:
    """Build a feature matrix from CSV upload or JSON rows."""
    expected = list(meta.get("feature_columns") or [])
    if csv_path is not None:
        columns = feature_columns or expected
        if not columns:
            raise ValueError("feature_columns required when model metadata has none")
        X, _, names = load_numeric_csv(csv_path, feature_columns=columns)
        return X, list(names)

    if not rows_json:
        raise ValueError("Provide a CSV file or JSON rows")
    try:
        rows = json.loads(rows_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON rows: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty JSON array")

    columns = feature_columns or expected
    if not columns:
        raise ValueError("feature_columns required when model metadata has none")

    if isinstance(rows[0], dict):
        missing = [c for c in columns if c not in rows[0]]
        if missing:
            raise ValueError(f"JSON rows missing columns: {missing}")
        matrix = [[float(row[c]) for c in columns] for row in rows]
    elif isinstance(rows[0], (list, tuple)):
        matrix = [[float(v) for v in row] for row in rows]
        if any(len(row) != len(columns) for row in matrix):
            raise ValueError(
                f"Each row must have {len(columns)} values matching feature_columns"
            )
    else:
        raise ValueError("rows must be an array of objects or arrays")

    return np.asarray(matrix, dtype=float), columns


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
    model_id = job_id
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
        model_path=job_dir / "model.npz",
    )

    components_local = Path(result["artifacts"]["components"])
    bmu_local = Path(result["artifacts"]["bmu"])
    model_local = Path(result["artifacts"]["model"])
    keys = _model_object_keys(model_id)

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = {
        "model_id": model_id,
        "job_id": job_id,
        "requested_by": requested_by,
        "created_at": created_at,
        "n_samples": result["n_samples"],
        "n_features": result["n_features"],
        "feature_columns": result["feature_columns"],
        "label_column": result["label_column"],
        "map_size": result["map_size"],
        "n_iterations": result["n_iterations"],
        "online": result["online"],
        "quantization_error": result["quantization_error"],
        "topographic_error": result["topographic_error"],
        "history": result["history"],
        "weights_shape": result["weights_shape"],
        "version": "1.0.0",
    }
    meta_path = job_dir / "meta.json"
    _write_meta_json(meta_path, meta)

    artifacts = {
        "components": upload_artifact(
            components_local,
            f"{job_id}/som_components.png",
            content_type="image/png",
        ),
        "bmu": upload_artifact(
            bmu_local,
            f"{job_id}/som_bmu.png",
            content_type="image/png",
        ),
        "model": upload_artifact(
            model_local,
            keys["model"],
            content_type="application/octet-stream",
        ),
        "meta": upload_artifact(
            meta_path,
            keys["meta"],
            content_type="application/json",
        ),
    }
    return {
        "job_id": job_id,
        "model_id": model_id,
        "requested_by": requested_by,
        "n_samples": result["n_samples"],
        "n_features": result["n_features"],
        "feature_columns": result["feature_columns"],
        "label_column": result["label_column"],
        "map_size": result["map_size"],
        "n_iterations": result["n_iterations"],
        "online": result["online"],
        "quantization_error": result["quantization_error"],
        "topographic_error": result["topographic_error"],
        "history": result["history"],
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
    MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
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


@app.get("/som/models/{model_id}")
def get_model(
    model_id: str,
    user: Annotated[dict[str, Any], Depends(verify_token)],
):
    """Return persisted model metadata from MinIO."""
    _ = user
    try:
        _, meta = _download_model_artifacts(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except S3Error as exc:
        raise HTTPException(
            status_code=502, detail=f"MinIO download failed: {exc}"
        ) from exc
    return meta


@app.post("/som/models/{model_id}/transform")
async def model_transform(
    model_id: str,
    user: Annotated[dict[str, Any], Depends(verify_token)],
    file: UploadFile | None = File(None, description="CSV with header row"),
    rows: str | None = Form(
        None,
        description='JSON array of objects or arrays, e.g. [{"a":1,"b":2}]',
    ),
    feature_columns: str | None = Form(
        None,
        description="Optional override; defaults to columns stored with the model",
    ),
):
    """Map samples to BMU coordinates using a persisted model."""
    _ = user
    return await _run_inference(
        model_id=model_id,
        file=file,
        rows=rows,
        feature_columns=feature_columns,
        mode="transform",
    )


@app.post("/som/models/{model_id}/predict")
async def model_predict(
    model_id: str,
    user: Annotated[dict[str, Any], Depends(verify_token)],
    file: UploadFile | None = File(None, description="CSV with header row"),
    rows: str | None = Form(
        None,
        description='JSON array of objects or arrays, e.g. [{"a":1,"b":2}]',
    ),
    feature_columns: str | None = Form(
        None,
        description="Optional override; defaults to columns stored with the model",
    ),
):
    """Map samples to flat BMU indices using a persisted model."""
    _ = user
    return await _run_inference(
        model_id=model_id,
        file=file,
        rows=rows,
        feature_columns=feature_columns,
        mode="predict",
    )


async def _run_inference(
    *,
    model_id: str,
    file: UploadFile | None,
    rows: str | None,
    feature_columns: str | None,
    mode: str,
) -> dict[str, Any]:
    columns = _parse_feature_columns(feature_columns) if feature_columns else None
    tmp_csv: Path | None = None
    if file is not None and file.filename:
        infer_dir = ARTIFACT_ROOT / "infer" / uuid.uuid4().hex[:12]
        infer_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename).suffix or ".csv"
        tmp_csv = infer_dir / f"input{suffix}"
        try:
            with tmp_csv.open("wb") as out:
                shutil.copyfileobj(file.file, out)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400, detail=f"Failed to store upload: {exc}"
            ) from exc

    if tmp_csv is None and not rows:
        raise HTTPException(
            status_code=400,
            detail="Provide a CSV file upload or Form field 'rows' (JSON)",
        )

    loop = asyncio.get_running_loop()

    def _infer() -> dict[str, Any]:
        model_path, meta = _download_model_artifacts(model_id)
        som = SOM.load(model_path)
        X, used_columns = _load_inference_matrix(
            csv_path=tmp_csv,
            rows_json=rows,
            feature_columns=columns,
            meta=meta,
        )
        if mode == "predict":
            values = som.predict(X).tolist()
            key = "bmu_indices"
        else:
            values = som.transform(X).tolist()
            key = "bmu_coords"
        return {
            "model_id": model_id,
            "n_samples": int(X.shape[0]),
            "feature_columns": used_columns,
            "map_size": [som.width, som.height],
            key: values,
        }

    try:
        return await loop.run_in_executor(None, _infer)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except S3Error as exc:
        raise HTTPException(
            status_code=502, detail=f"MinIO download failed: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
