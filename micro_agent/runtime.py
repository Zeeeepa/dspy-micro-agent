from __future__ import annotations
import json, os, re, time, uuid, datetime
from typing import Any, Dict, List, Optional, TypedDict, NotRequired
import ast
try:
    import json_repair
except Exception:
    json_repair = None

TRACES_DIR = os.getenv("TRACES_DIR", "traces")
os.makedirs(TRACES_DIR, exist_ok=True)

def _get_traces_dir() -> str:
    return os.getenv("TRACES_DIR", TRACES_DIR)

class Step(TypedDict):
    tool: str
    args: Dict[str, Any]
    observation: Any

class TraceRecord(TypedDict):
    id: str
    ts: str
    question: str
    steps: List[Step]
    answer: str
    usage: NotRequired[Dict[str, Any]]
    cost_usd: NotRequired[float]

def new_trace_id() -> str:
    return uuid.uuid4().hex

def dump_trace(trace_id: str, question: str, steps: List[Step], answer: str, *, usage: Optional[Dict[str, Any]] = None, cost_usd: Optional[float] = None) -> str:
    rec: TraceRecord = {
        "id": trace_id,
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "steps": steps,
        "answer": answer,
    }
    if usage is not None:
        rec["usage"] = usage
    if cost_usd is not None:
        rec["cost_usd"] = float(cost_usd)
    traces_dir = _get_traces_dir()
    os.makedirs(traces_dir, exist_ok=True)
    path = os.path.join(traces_dir, f"{trace_id}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return path

def extract_json_block(text: str) -> str:
    """
    Extract the first {...} block to survive models adding prose or code fences.
    """
    if not text:
        raise ValueError("No JSON object found in empty text")
    start = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
            continue
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError(f"No JSON object found in: {text[:200]!r}")

def parse_decision_text(text: str) -> Dict[str, Any]:
    """Parse a model decision string into a dict.

    Strategy:
    1) Extract first {...} block.
    2) Try strict JSON parse.
    3) Fallback to Python literal parse (single quotes, etc.).
    """
    block = extract_json_block(text)
    # 1) strict json
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Decision JSON is not an object")
    except Exception:
        pass
    # 2) json-repair (if available)
    if json_repair is not None:
        try:
            repaired = json_repair.repair(block)
            if isinstance(repaired, dict):
                return repaired
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
            raise ValueError("Decision JSON is not an object")
        except Exception:
            pass
    # 3) python literal (handles single quotes)
    try:
        obj = ast.literal_eval(block)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    raise ValueError("Could not parse decision as JSON or Python literal")

def now_iso(utc: bool = False) -> str:
    if utc:
        return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return datetime.datetime.now().isoformat(timespec="seconds")
