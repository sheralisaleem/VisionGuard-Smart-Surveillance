from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
from detection.stream_processor import StreamProcessor
from pathlib import Path
import shutil
import os

# Initialize the FastAPI app
app = FastAPI(title="Smart Surveillance API")

# Setup base directory paths to avoid E: drive pathing issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Initialize our AI Vision Engine
print("Initializing AI Engine...")
processor = StreamProcessor(source=0)

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the main HTML dashboard."""
    html_path = os.path.join(BASE_DIR, "frontend", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/video_feed")
def video_feed():
    """Streams the AI-processed video feed to the web browser."""
    return StreamingResponse(processor.generate_frames(), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/get_logs")
def get_logs():
    """Returns the latest event logs from the Vision Engine."""
    return {"logs": processor.get_logs()}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Ensure the uploads folder exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            
        # Create a clean absolute path for the file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Tell the processor to switch to this file
        processor.change_source(file_path)
        
        print(f"✅ Video Uploaded Successfully: {file_path}")
        return {"message": f"Successfully uploaded {file.filename}. Switching to file stream!"}
    
    except Exception as e:
        print(f"❌ Upload Error: {str(e)}")
        return {"message": f"Server Error: {str(e)}"}, 500

@app.on_event("shutdown")
def shutdown_event():
    """Cleans up the alerts folder when the server shuts down."""
    print("🧹 Cleaning up alerts folder...")
    alerts_dir = os.path.join(BASE_DIR, "alerts")
    if os.path.exists(alerts_dir):
        for filename in os.listdir(alerts_dir):
            file_path = os.path.join(alerts_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Failed to delete {file_path}. Reason: {e}")
        print("✅ Alerts folder cleared successfully.")

# Run the server
if __name__ == "__main__":
    print("🚀 Starting FastAPI Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)