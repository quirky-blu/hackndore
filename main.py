from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

app = FastAPI(title="Streetlight Points & Citizen Report API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure OpenAI Configuration
AZURE_ENDPOINT = "https://models.github.ai/inference"
GPT_MODEL = "openai/gpt-4.1"

# Load environment variables
load_dotenv()
current_dir = Path(__file__).parent
env_file = current_dir / '.env'
load_dotenv(env_file)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Load GeoJSON data at startup
geojson_data = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    query_type: str = "chat"
    files_to_query: List[str] = []
    response_description: str = ""
    search_terms: List[str] = []

@app.on_event("startup")
async def load_geojson_data():
    global geojson_data
    try:
        with open('ring_roads_streetlight_points.geojson', 'r') as f:
            geojson_data = json.load(f)
        print(f"Loaded {len(geojson_data.get('features', []))} features from GeoJSON file")
    except FileNotFoundError:
        print("GeoJSON file not found. Creating empty feature collection.")
        geojson_data = {"type": "FeatureCollection", "features": []}
    except json.JSONDecodeError:
        print("Error parsing GeoJSON file. Creating empty feature collection.")
        geojson_data = {"type": "FeatureCollection", "features": []}

async def query_gpt4(prompt: str, available_files: List[str] = []) -> Dict[str, Any]:
       
    system_prompt = f"""
You are an expert in analyzing data from smart streetlamp IoT systems. These streetlamps collect data on air pollution (PM2.5, PM10, CO2, NO2, VOCs) and noise pollution (dB levels, time patterns, peak sources). 

Your role is to:
- Interpret noise and air quality data clearly and accurately. 
- Highlight health, safety, and lifestyle impacts for residents based on collected IoT data.
- Provide insights into trends such as high traffic noise, industrial emissions, or evening nightlife disruptions.
- Recommend whether the user should consider buying or renting a house in that area, based on environmental quality indicators.
- Keep answers practical, evidence-based, and easy to understand, as if advising a potential homebuyer.
- Assume that a road named ring road has high pollution and high noise therefore not recommanded to buy a house there

If the user asks non-related questions, politely redirect to IoT streetlamp environmental analysis topics.
"""

    try:
        if not GITHUB_TOKEN:
            raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured")
            
        client = ChatCompletionsClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(GITHUB_TOKEN),
        )
        
        # Use dictionary format for messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = client.complete(
            messages=messages,
            temperature=0.1,
            model=GPT_MODEL
        )
        
        ai_response = response.choices[0].message.content
        
        # Try to parse JSON response, if not return as plain text
        try:
            parsed_response = json.loads(ai_response)
            return parsed_response
        except json.JSONDecodeError:
            # If not JSON, treat as regular chat response
            return {
                "query_type": "chat",
                "response": ai_response,
                "files_to_query": available_files[:3] if available_files else [],
                "response_description": "Chat response from citizen report bot",
                "search_terms": []
            }
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying GPT-4 API: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Streetlight Points & Citizen Report API", 
        "version": "1.0.0",
        "endpoints": {
            "/api/points": "GET - Fetch points by bounding box",
            "/api/chat": "POST - Chat with citizen report bot",
            "/api/health": "GET - Health check"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "features_loaded": len(geojson_data.get('features', [])) if geojson_data else 0,
        "gpt_configured": GITHUB_TOKEN is not None
    }

@app.post("/api/chat")
async def chat_with_bot(request: ChatRequest) -> ChatResponse:
    """
    Chat with the citizen report bot for NAVBHARAT.
    
    Args:
        request: ChatRequest containing the user message
        
    Returns:
        ChatResponse with bot's reply and metadata
    """
    try:
        # Get available files (you can modify this based on your needs)
        available_files = ["streetlight_data.json", "road_reports.json", "maintenance_logs.json"]
        
        # Query GPT-4
        gpt_response = await query_gpt4(request.message, available_files)
        
        # Format response
        return ChatResponse(
            response=gpt_response.get("response", gpt_response.get("content", "Sorry, I couldn't process your request.")),
            query_type=gpt_response.get("query_type", "chat"),
            files_to_query=gpt_response.get("files_to_query", []),
            response_description=gpt_response.get("response_description", ""),
            search_terms=gpt_response.get("search_terms", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat processing: {str(e)}")

@app.get("/api/points")
async def get_points(
    north: float = Query(..., description="Northern latitude boundary", ge=-90, le=90),
    south: float = Query(..., description="Southern latitude boundary", ge=-90, le=90),
    east: float = Query(..., description="Eastern longitude boundary", ge=-180, le=180),
    west: float = Query(..., description="Western longitude boundary", ge=-180, le=180)
) -> Dict[str, Any]:
    """
    Fetch streetlight points within the specified bounding box.
    
    Args:
        north: Northern latitude boundary
        south: Southern latitude boundary  
        east: Eastern longitude boundary
        west: Western longitude boundary
        
    Returns:
        GeoJSON FeatureCollection with points in the bounding box
    """
    
    if not geojson_data:
        return {"type": "FeatureCollection", "features": []}
    
    # Validate bounding box
    if south > north:
        return {"error": "South latitude must be less than north latitude"}
    if west > east:
        return {"error": "West longitude must be less than east longitude"}
    
    # Filter points within the bounding box
    filtered_features = []
    
    for feature in geojson_data.get('features', []):
        if feature.get('geometry', {}).get('type') == 'Point':
            coordinates = feature.get('geometry', {}).get('coordinates', [])
            if len(coordinates) >= 2:
                lon, lat = coordinates[0], coordinates[1]
                
                # Check if point is within bounding box
                if (south <= lat <= north) and (west <= lon <= east):
                    filtered_features.append(feature)
    
    return {
        "type": "FeatureCollection", 
        "features": filtered_features,
        "count": len(filtered_features),
        "bounding_box": {
            "north": north,
            "south": south,
            "east": east,
            "west": west
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


