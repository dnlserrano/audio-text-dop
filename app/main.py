import os
import uuid
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.transcriber import WhisperTranscriber

# Create FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="API for transcribing audio files using OpenAI's Whisper model",
    version="1.0.0"
)

# Setup Jinja2 templates
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(Path(BASE_DIR) / "app" / "templates"))

# Mount static files
app.mount("/static", StaticFiles(directory=str(Path(BASE_DIR) / "static")), name="static")

# Create temp directory if it doesn't exist
TEMP_DIR = Path(BASE_DIR) / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Maintain a dictionary of active transcription jobs
active_jobs = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with file upload form"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    model_size: str = "large",
    language: str = "pt",
    gpu_backend: str = "auto",
    chunk_duration: int = 30,
):
    """
    Upload an audio file and start transcription process.
    Returns a job ID that can be used to stream results.
    """
    # Save the uploaded file to a temporary location
    temp_file_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Create a job ID
    job_id = str(uuid.uuid4())

    # Create a transcriber instance with user parameters
    transcriber = WhisperTranscriber(
        model_size=model_size,
        language=language,
        gpu_backend=gpu_backend,
        chunk_duration=chunk_duration
    )

    # Store the job details
    active_jobs[job_id] = {
        "file_path": str(temp_file_path),
        "transcriber": transcriber,
        "status": "ready"
    }

    return {"job_id": job_id, "filename": file.filename, "status": "ready"}

async def generate_stream(job_id: str):
    """Generator function that yields transcription results as they come in"""
    if job_id not in active_jobs:
        yield "Event: error\nData: Job not found\n\n"
        return

    job = active_jobs[job_id]
    job["status"] = "processing"

    try:
        # Send initial message
        yield "event: start\ndata: Transcription started\n\n"

        # Stream transcription results
        async for paragraph in job["transcriber"].transcribe_stream(job["file_path"]):
            # Format for server-sent events
            yield f"event: transcription\ndata: {paragraph}\n\n"

        # Send completion message
        yield "event: complete\ndata: Transcription completed\n\n"

        # Update job status
        job["status"] = "completed"
    except Exception as e:
        # Handle errors
        error_message = str(e)
        yield f"event: error\ndata: {error_message}\n\n"
        job["status"] = "error"
    finally:
        # Clean up the temporary file
        if os.path.exists(job["file_path"]):
            os.remove(job["file_path"])

@app.get("/stream/{job_id}")
async def stream_results(job_id: str):
    """Stream transcription results using server-sent events"""
    if job_id not in active_jobs:
        return {"error": "Job not found"}

    return StreamingResponse(
        generate_stream(job_id),
        media_type="text/event-stream"
    )

@app.get("/status/{job_id}")
async def job_status(job_id: str):
    """Get the status of a transcription job"""
    if job_id not in active_jobs:
        return {"error": "Job not found"}

    return {
        "job_id": job_id,
        "status": active_jobs[job_id]["status"]
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and clean up resources"""
    if job_id not in active_jobs:
        return {"error": "Job not found"}

    # Clean up the temporary file if it exists
    file_path = active_jobs[job_id]["file_path"]
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove the job from active jobs
    del active_jobs[job_id]

    return {"status": "deleted", "job_id": job_id}

@app.on_event("shutdown")
def cleanup():
    """Clean up temporary files when shutting down"""
    for job_id, job in active_jobs.items():
        if os.path.exists(job["file_path"]):
            os.remove(job["file_path"])
