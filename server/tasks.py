# tasks.py
import multiprocessing
import time
from datetime import datetime, timedelta

# Storage for running tasks
TASKS = {}  # task_id -> {"process": Process, "result": None, "start_time": datetime, "session_id": str}

# -----------------------------
# Helper: generate unique task IDs
# -----------------------------
import uuid
def generate_task_id():
    return str(uuid.uuid4())

# -----------------------------
# Start a background task
# -----------------------------
def start_task(request_data, session_id=None):
    from app import detect_text_logic 
    task_id = generate_task_id()

    def task_wrapper():
        try:
            result = detect_text_logic(request_data) 
            TASKS[task_id]["result"] = result
        except Exception as e:
            TASKS[task_id]["result"] = {"error": str(e)}

    p = multiprocessing.Process(target=task_wrapper)
    p.start()

    TASKS[task_id] = {
        "process": p, 
        "result": None, 
        "start_time": datetime.utcnow(),
        "session_id": session_id
    }
    return task_id

def task_running(task_id):
    task = TASKS.get(task_id)
    if not task:
        return False
    return task["process"].is_alive()

def get_task_result(task_id):
    task = TASKS.get(task_id)
    if task and task["result"] is not None:
        return task["result"]
    return None

def cancel_task(task_id):
    task = TASKS.get(task_id)
    if task and task["process"].is_alive():
        task["process"].terminate()
        task["process"].join()
        task["result"] = {"status": "cancelled"}
        return True
    return False

def cancel_session_tasks(session_id):
    """Cancel all running tasks for a specific session"""
    if not session_id:
        return {"cancelled": 0, "message": "No session_id provided"}
    
    cancelled_count = 0
    cancelled_tasks = []
    
    for tid, task in list(TASKS.items()):
        if task.get("session_id") == session_id and task["process"].is_alive():
            try:
                task["process"].terminate()
                task["process"].join(timeout=2)  
                task["result"] = {"status": "cancelled", "reason": "user_exit"}
                cancelled_count += 1
                cancelled_tasks.append(tid)
            except Exception as e:
                print(f"Error cancelling task {tid}: {e}")
    
    return {
        "cancelled": cancelled_count,
        "task_ids": cancelled_tasks,
        "message": f"Cancelled {cancelled_count} tasks for session {session_id}"
    }

def cleanup_expired_tasks(max_age_minutes=30):
    now = datetime.utcnow()
    removed = []
    for tid, task in list(TASKS.items()):
        if now - task["start_time"] > timedelta(minutes=max_age_minutes):
            if task["process"].is_alive():
                task["process"].terminate()
                task["process"].join()
            TASKS.pop(tid, None)
            removed.append(tid)
    return removed

def get_session_tasks(session_id):
    """Get all active task IDs for a session"""
    if not session_id:
        return []
    
    return [
        tid for tid, task in TASKS.items() 
        if task.get("session_id") == session_id and task["process"].is_alive()
    ]