from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
import subprocess

app = FastAPI(title="jarvis-runner")

ALLOWED_ROOT = Path("/opt/jarvis").resolve()

class CmdResult(BaseModel):
    ok: bool
    exit_code: int
    stdout: str
    stderr: str

@app.get("/runner/health")
def health():
    return {"status": "ok", "service": "jarvis-runner"}

@app.get("/runner/fs/list", response_model=CmdResult)
def fs_list(path: str = Query("/opt/jarvis", description="Directory to list under /opt/jarvis")):
    try:
        target = Path(path).resolve()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}")

    # Must be inside /opt/jarvis
    if target != ALLOWED_ROOT and ALLOWED_ROOT not in target.parents:
        raise HTTPException(status_code=403, detail="Path not allowed")

    if not target.exists():
        raise HTTPException(status_code=404, detail="Path does not exist")

    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    # No shell. Fixed command. Read-only.
    proc = subprocess.run(
        ["ls", "-la", str(target)],
        capture_output=True,
        text=True,
    )

    return CmdResult(
        ok=(proc.returncode == 0),
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
