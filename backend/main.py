"""
FastAPI Backend for Emotion-Aware NPCs
Provides emotion inference API endpoints for Unity client
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time
import os
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion-Aware NPCs API",
    description="Real-time emotion inference for adaptive NPC dialogue",
    version="1.0.0"
)

# CORS middleware for Unity client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    timestamp: float
    features: Dict[str, Any]

class FrameData(BaseModel):
    frame_data: str  # Base64 encoded image data
    timestamp: float

# Global state for latest emotion (stub implementation)
latest_emotion = {
    "emotion": "neutral",
    "confidence": 0.85,
    "timestamp": time.time(),
    "features": {
        "valence": 0.0,
        "arousal": 0.0,
        "stress_level": 0.2,
        "fatigue_level": 0.1,
        "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "eye_aspect_ratio": 0.25,
        "motion_intensity": 0.05
    }
}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Emotion-Aware NPCs API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "infer": "/infer",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "emotion-inference-api"
    }

@app.get("/infer", response_model=EmotionResponse)
async def get_latest_emotion():
    """
    Get the latest emotion prediction
    Returns the most recent emotion inference result
    """
    try:
        logger.info(f"Returning latest emotion: {latest_emotion['emotion']}")
        return EmotionResponse(**latest_emotion)
    except Exception as e:
        logger.error(f"Error getting latest emotion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/infer", response_model=EmotionResponse)
async def process_frame(frame_data: FrameData):
    """
    Process a new frame and return emotion prediction
    This is a stub implementation until the actual model is connected
    """
    try:
        # TODO: Replace with actual model inference
        # For now, return a mock response with slight variations
        
        # Simulate processing time
        time.sleep(0.05)  # 50ms processing time
        
        # Mock emotion prediction with some variation
        emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
        import random
        emotion = random.choice(emotions)
        confidence = round(random.uniform(0.7, 0.95), 2)
        
        # Update global state
        global latest_emotion
        latest_emotion = {
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": time.time(),
            "features": {
                "valence": round(random.uniform(-1.0, 1.0), 2),
                "arousal": round(random.uniform(0.0, 1.0), 2),
                "stress_level": round(random.uniform(0.0, 1.0), 2),
                "fatigue_level": round(random.uniform(0.0, 0.5), 2),
                "head_pose": {
                    "yaw": round(random.uniform(-30, 30), 1),
                    "pitch": round(random.uniform(-20, 20), 1),
                    "roll": round(random.uniform(-10, 10), 1)
                },
                "eye_aspect_ratio": round(random.uniform(0.2, 0.3), 2),
                "motion_intensity": round(random.uniform(0.0, 0.2), 2)
            }
        }
        
        logger.info(f"Processed frame, predicted emotion: {emotion} (confidence: {confidence})")
        return EmotionResponse(**latest_emotion)
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail="Error processing frame")

@app.get("/status")
async def get_status():
    """Get system status and performance metrics"""
    return {
        "status": "running",
        "uptime": time.time(),
        "latest_emotion": latest_emotion,
        "performance": {
            "target_latency_ms": 400,
            "target_fps": 15,
            "current_processing_time_ms": 50
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Emotion-Aware NPCs API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
