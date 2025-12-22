from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, json, re
from threading import Lock
from pydantic import BaseModel
from .costs import estimate_prediction_cost
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from .config import configure_lm
from .agent import MicroAgent
from .runtime import dump_trace, new_trace_id, to_jsonable
from .logging_setup import setup_logging

app = FastAPI(title="DSPy Micro Agent")
origins_env = os.getenv("MICRO_AGENT_CORS_ORIGINS", "*").strip()
if origins_env == "*":
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    allow_credentials = os.getenv("MICRO_AGENT_CORS_CREDENTIALS", "0").strip().lower() in {"1", "true", "yes", "on"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    max_steps: int = 6
    use_tool_calls: bool | None = None

class AskResponse(BaseModel):
    answer: str
    trace_id: str
    trace_path: str
    steps: list
    usage: dict | None = None
    cost_usd: float | None = None

setup_logging()
configure_lm()
_agent = MicroAgent()
_agent_lock = Lock()
_serialize = os.getenv("MICRO_AGENT_SERIALIZE", "1").strip().lower() not in {"0", "false", "no", "off"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/health")
def health():
    lm = _agent.lm
    return {
        "status": "ok",
        "provider": getattr(_agent, "_provider", None),
        "model": getattr(lm, "model", None),
        "max_steps": getattr(_agent, "max_steps", None),
    }

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must be a non-empty string")
    if req.max_steps < 1 or req.max_steps > 20:
        raise HTTPException(status_code=400, detail="max_steps must be between 1 and 20")

    def _call_agent():
        if _serialize and req.use_tool_calls is None and req.max_steps == _agent.max_steps:
            return _agent(question)
        agent = MicroAgent(max_steps=req.max_steps, use_tool_calls=req.use_tool_calls, use_global_trace=_serialize)
        return agent(question)

    if _serialize:
        with _agent_lock:
            pred = _call_agent()
    else:
        pred = _call_agent()

    trace_id = new_trace_id()
    usage = getattr(pred, "usage", {}) or {}
    est = estimate_prediction_cost(question, pred.trace, pred.answer, usage)
    path = dump_trace(trace_id, question, pred.trace, pred.answer, usage=usage, cost_usd=est.get("cost_usd"))
    steps = to_jsonable(pred.trace)
    return AskResponse(answer=pred.answer, trace_id=trace_id, trace_path=path, steps=steps, usage=usage, cost_usd=est.get("cost_usd"))

@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id, flags=re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid trace id format")
    traces_dir = os.getenv("TRACES_DIR", "traces")
    path = os.path.join(traces_dir, f"{trace_id}.jsonl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Trace not found")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return {"trace_id": trace_id, "path": path, "records": out}

@app.get("/version")
def api_version():
    try:
        ver = _pkg_version("dspy-micro-agent")
    except PackageNotFoundError:
        ver = "0.0.0"
    return {"name": "dspy-micro-agent", "version": ver}
