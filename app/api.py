import os
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the core recovery pipeline router
from pipeline.step6_route.service import route

# Create the FastAPI app
app = FastAPI(
    title="Recovery AI Firewall",
    description="Real-time prompt injection defense middleware for LLM applications.",
    version="0.2.0"
)

# ── Pydantic Schemas ─────────────────────────────────────────────────────────
class CheckPromptRequest(BaseModel):
    prompt: str

class CheckPromptResponse(BaseModel):
    final_route: str
    reason: str
    prompt_used: Optional[str]
    total_tokens: int
    latency_ms: float
    audit_log: Dict[str, Any]

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Simple health check to verify the WAF is running."""
    return {"status": "ok", "version": "0.2.0"}

@app.post("/v1/check", response_model=CheckPromptResponse)
async def check_prompt(
    request: CheckPromptRequest,
    x_api_key: Optional[str] = Header(None, description="Optional Groq API Key to override the server's default")
):
    """
    Evaluates a raw user prompt against the Recovery 6-Stage Defense Pipeline.
    
    Returns the routing decision ("production" or "reject") along with the 
    safe prompt to use (which may be the original prompt or a sanitized version).
    """
    try:
        # Pass the prompt into the router
        result = route(
            user_input=request.prompt,
            api_key=x_api_key or os.getenv("GROQ_API_KEY")
        )
        
        return {
            "final_route": result["final_route"],
            "reason": result["reason"],
            "prompt_used": result["prompt_used"],
            "total_tokens": result["total_tokens"],
            "latency_ms": result["latency_ms"],
            "audit_log": result["audit_log"]
        }
        
    except Exception as e:
        # Fail-safe open or closed? Here we fail-closed on internal errors.
        raise HTTPException(status_code=500, detail=str(e))

# Middleware to automatically inject the X-Recovery-Latency header
@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Recovery-Latency-Ms"] = f"{process_time_ms:.2f}"
    return response

if __name__ == "__main__":
    import uvicorn
    # When run directly, start the uvicorn server on port 8000
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
