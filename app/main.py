import os
import uuid
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import json
import os
import tempfile
import uuid
from pathlib import Path

import redis
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

TEMP_DIR = Path(tempfile.gettempdir())

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")  # Use "redis" to resolve the redis service name
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Create a Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


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

    # Store the job details in Redis
    job_data = {
        "file_path": str(temp_file_path),
        "transcriber": {
            "model_size": transcriber.model_size,
            "language": transcriber.language,
            "gpu_backend": transcriber.gpu_backend,
            "chunk_duration": transcriber.chunk_duration
        },
        "status": "ready"
    }
    redis_client.set(job_id, json.dumps(job_data))

    return {"job_id": job_id, "filename": file.filename, "status": "ready"}


async def generate_stream(job_id: str):
    """Generator function that yields transcription results as they come in"""
    job_data_str = redis_client.get(job_id)
    if not job_data_str:
        yield "Event: error\nData: Job not found\n\n"
        return

    job_data = json.loads(job_data_str)
    job_data["status"] = "processing"
    redis_client.set(job_id, json.dumps(job_data))

    # Re-create the transcriber instance
    transcriber = WhisperTranscriber(
        model_size=job_data["transcriber"]["model_size"],
        language=job_data["transcriber"]["language"],
        gpu_backend=job_data["transcriber"]["gpu_backend"],
        chunk_duration=job_data["transcriber"]["chunk_duration"]
    )

    try:
        # Send initial message
        yield "event: start\ndata: Transcription started\n\n"

        # Stream transcription results
        async for paragraph in transcriber.transcribe_stream(job_data["file_path"]):
            # Format for server-sent events
            yield f"event: transcription\ndata: {paragraph}\n\n"

        # Send completion message
        yield "event: complete\ndata: Transcription completed\n\n"

        # Update job status
        job_data["status"] = "completed"
        redis_client.set(job_id, json.dumps(job_data))

    except Exception as e:
        # Handle errors
        error_message = str(e)
        yield f"event: error\ndata: {error_message}\n\n"
        job_data["status"] = "error"
        redis_client.set(job_id, json.dumps(job_data))
    finally:
        # Clean up the temporary file
        if os.path.exists(job_data["file_path"]):
            os.remove(job_data["file_path"])


@app.get("/stream/{job_id}")
async def stream_results(job_id: str):
    """Stream transcription results using server-sent events"""
    job_data_str = redis_client.get(job_id)
    if not job_data_str:
        return {"error": "Job not found"}

    return StreamingResponse(
        generate_stream(job_id),
        media_type="text/event-stream"
    )


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    """Get the status of a transcription job"""
    job_data_str = redis_client.get(job_id)
    if not job_data_str:
        return {"error": "Job not found"}

    job_data = json.loads(job_data_str)
    return {
        "job_id": job_id,
        "status": job_data["status"]
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and clean up resources"""
    job_data_str = redis_client.get(job_id)
    if not job_data_str:
        return {"error": "Job not found"}

    job_data = json.loads(job_data_str)

    # Clean up the temporary file if it exists
    file_path = job_data["file_path"]
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove the job from Redis
    redis_client.delete(job_id)

    return {"status": "deleted", "job_id": job_id}


@app.on_event("shutdown")
def cleanup():
    """Clean up temporary files when shutting down"""
    # Iterate through all job keys in Redis
    for job_id in redis_client.scan_iter():
        job_data_str = redis_client.get(job_id)
        if job_data_str:
            job_data = json.loads(job_data_str)
            if os.path.exists(job_data["file_path"]):
                os.remove(job_data["file_path"])
            redis_client.delete(job_id)
