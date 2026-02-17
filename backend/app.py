import os
import uuid
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.background import BackgroundTasks

from analytics import process_video_job

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="WareSight Movement Analytics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(RUNS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, "input.mp4")

    # Save upload
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Init job state
    JOBS[job_id] = {
        "status": "uploaded",
        "progress": 5,
        "message": "Uploaded",
        "job_dir": job_dir,
        "outputs": {},
        "error": None,
    }

    # Run async in background
    background_tasks.add_task(process_video_job, job_id, JOBS)

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    data = JOBS[job_id].copy()
    # don't expose server paths
    data.pop("job_dir", None)
    return data


@app.get("/api/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    if job_id not in JOBS:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    job_dir = JOBS[job_id]["job_dir"]
    path = os.path.join(job_dir, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(path, filename=filename)
