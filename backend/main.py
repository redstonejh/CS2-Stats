from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
from demo_parser.parser import DemoParser

app = FastAPI(title="CS2 Stats API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-demo")
async def analyze_demo(demo: UploadFile = File(...)):
    print(f"Received demo file: {demo.filename}")
    
    # Save uploaded file
    temp_demo_path = f"temp_{demo.filename}"
    with open(temp_demo_path, "wb") as buffer:
        shutil.copyfileobj(demo.file, buffer)
    
    try:
        # Parse demo
        parser = DemoParser(temp_demo_path)
        analysis = parser.parse()
        
        return {
            "success": True,
            "data": analysis
        }
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise
    finally:
        # Cleanup
        os.remove(temp_demo_path)

@app.get("/")
async def root():
    return {"message": "CS2 Stats API is running"}