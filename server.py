from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from environment import CustomerSupportEnv, Action
from tasks import get_grader, TASKS

app = FastAPI(title="Customer Support Triage Environment")

# Global environment instance
env = CustomerSupportEnv(task_level="easy")

# ─── Request Models ────────────────────────────────────────

class ActionRequest(BaseModel):
    action_type: str
    value: str
    task_level: Optional[str] = "easy"

class ResetRequest(BaseModel):
    task_level: Optional[str] = "easy"

# ─── Endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "message": "Customer Support Triage Environment is running"}

@app.post("/reset")
def reset(request: ResetRequest):
    global env
    task_level = request.task_level or "easy"
    if task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="task_level must be easy, medium, or hard")
    env = CustomerSupportEnv(task_level=task_level)
    obs = env.reset()
    return {
        "observation": obs.dict(),
        "task": TASKS[task_level],
        "message": f"Environment reset for {task_level} task"
    }

@app.post("/step")
def step(request: ActionRequest):
    global env
    if env.current_ticket is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    action = Action(
        action_type=request.action_type,
        value=request.value
    )
    result = env.step(action)
    grader = get_grader(env.task_level)
    final_score = grader(env) if result.done else None
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
        "final_score": final_score
    }

@app.get("/state")
def state():
    global env
    return env.state()

@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS}

@app.get("/")
def root():
    return {
        "name": "Customer Support Triage Environment",
        "version": "1.0.0",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks"],
        "tasks": ["easy", "medium", "hard"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)