import os
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Create the FastAPI app
app = FastAPI(
    title="Recovery AI Firewall",
    description="Real-time prompt injection detection + repair (6-stage pipeline)",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS - essential for research/demo use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional rate limiting (uncomment if needed for public research API)
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter
# app.add_exception_handler(429, _rate_limit_exceeded_handler)

# ── Pydantic Schemas ─────────────────────────────────────────────────────────
class CheckPromptRequest(BaseModel):
    prompt: str

class CheckPromptResponse(BaseModel):
    final_route: str
    reason: str
    prompt_used: Optional[str]
    total_tokens: int
    latency_ms: float
    stages: Dict[str, Any]      # Full audit trail (research gold)
    audit_log: Dict[str, Any]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check + readiness for deployment."""
    return {
        "status": "healthy",
        "version": "0.2.0",
        "note": "Run 'python scripts/run_step2.py' if models are missing"
    }


@app.post("/v1/check", response_model=CheckPromptResponse)
async def check_prompt(
    request: CheckPromptRequest,
    x_api_key: Optional[str] = Header(
        None, description="Optional Groq API key override"
    ),
):
    """
    Main endpoint: Run the full 6-stage recovery pipeline on a prompt.
    """
    start_time = time.perf_counter()  # For optional extra metrics

    # Lazy import → prevents startup crash on fresh clone
    try:
        from pipeline.step6_route.service import route
    except FileNotFoundError as e:
        # This catches the missing model artifacts gracefully
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run this once: "
                   "python scripts/run_step2.py"
        ) from e

    try:
        result = route(
            user_input=request.prompt,
            api_key=x_api_key or os.getenv("GROQ_API_KEY")
        )

        # Optional: log full request latency (middleware already adds header)
        # latency = (time.perf_counter() - start_time) * 1000

        return result   # ← structure already perfectly matches the schema

    except Exception as e:
        # Fail-closed for security (research best practice)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)[:200]}"
        ) from e


# Global middleware: adds X-Recovery-Latency-Ms header on every response
@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start) * 1000
    response.headers["X-Recovery-Latency-Ms"] = f"{process_time:.2f}"
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",          # ← works when saved as app/api.py
        host="0.0.0.0",
        port=8000,
        reload=True,             # great for research iteration
        log_level="info"
    )
