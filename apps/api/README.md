# API service

FastAPI application for SOM training, Keycloak JWT auth, and MinIO artifact upload.

```bash
# from monorepo root
pip install -e packages/som_core
pip install -r apps/api/requirements.txt
WEB_ROOT=apps/web uvicorn apps.api.main:app --app-dir apps/api --reload
# or:
cd apps/api && WEB_ROOT=../web uvicorn main:app --reload
```
