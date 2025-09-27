import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Workflow State (Mock State) ---
# ใน production ควรใช้ฐานข้อมูลจริง (เช่น Firestore)
workflow_status = {
    "isRunning": False,
    "currentStep": 0,
    "steps": [
        {"id": 1, "name": "Load Rubrics & QA", "status": "waiting", "progress": 0},
        {"id": 2, "name": "Ingest Evidence", "status": "waiting", "progress": 0},
        {"id": 3, "name": "Retrieve Contexts (RAG)", "status": "waiting", "progress": 0},
        {"id": 4, "name": "Generate Assessment Summary (LLM)", "status": "waiting", "progress": 0},
        {"id": 5, "name": "Finalize Results", "status": "waiting", "progress": 0},
    ]
}

# Mock storage for final assessment results
assessment_results = [] 

# -----------------------------------------------------------
# 1. Main Workflow Runner
# -----------------------------------------------------------

def run_assessment_workflow():
    """
    Simulates the 5-step RAG assessment process. 
    This is typically run as a background task.
    """
    global workflow_status
    global assessment_results
    
    if workflow_status["isRunning"]:
        logging.warning("Assessment workflow is already running.")
        return

    logging.info("Starting 5-step assessment workflow...")
    workflow_status["isRunning"] = True
    assessment_results = [] # Clear previous results

    try:
        # Step 1: Load documents (Stub)
        _update_step_status(1, 'running', 20)
        logging.info("Step 1: Rubrics and QAs loaded successfully.")
        
        # Step 2: Ingest Evidence (Stub)
        _update_step_status(2, 'running', 40)
        logging.info("Step 2: Evidence ingested and indexed.")
        
        # Step 3: RAG Retrieval (Stub)
        _update_step_status(3, 'running', 60)
        # Mocking retrieval result for one question
        mock_context = "หลักฐาน A ระบุว่ามีแผนการดำเนินงานปี 2567 ที่ครบถ้วน และหลักฐาน B ชี้ถึงช่องโหว่ในการติดตามผลในไตรมาสที่ 3"
        
        # Step 4: LLM Generation (Stub)
        _update_step_status(4, 'running', 80)
        # Assuming the LLM generated a summary for a test question
        mock_summary = {
            "question": "แผนการดำเนินงานปี 2567 มีความแข็งแกร่งหรือไม่?",
            "summary": "หลักฐาน A เป็น Strong evidence แต่หลักฐาน B เป็น Gap",
            "context_used": mock_context
        }
        assessment_results.append(mock_summary)

        # Step 5: Finalize (Stub)
        _update_step_status(5, 'done', 100)
        logging.info("Assessment workflow completed successfully.")

    except Exception as e:
        logging.error(f"Assessment workflow failed: {e}")
        _update_step_status(workflow_status["currentStep"], 'error', 0)
    finally:
        workflow_status["isRunning"] = False


# -----------------------------------------------------------
# 2. Status and Results Getters
# -----------------------------------------------------------

def get_workflow_status() -> Dict[str, Any]:
    """Returns the current status of the background assessment process."""
    return workflow_status

def get_workflow_results() -> List[Dict[str, Any]]:
    """Returns the final collected assessment results."""
    return assessment_results

# -----------------------------------------------------------
# 3. Utility Function
# -----------------------------------------------------------

def _update_step_status(step_id: int, status: str, progress: int):
    """Internal helper to update the status and progress."""
    global workflow_status
    workflow_status["currentStep"] = step_id
    for step in workflow_status["steps"]:
        if step["id"] == step_id:
            step["status"] = status
            step["progress"] = progress
            break
