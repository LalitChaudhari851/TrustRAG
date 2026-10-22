import logfire
from typing import List
from fastapi import APIRouter, Request, Response, HTTPException, status
from app.models.api_schemas import ThreadSummary, ThreadDetail, ThreadMessage
from app.models.feedback_db import get_user_threads, get_thread_detail, delete_user_thread
from app.api.auth import get_current_user

router = APIRouter(prefix="/v1", tags=["Threads"])


@router.get("/threads", response_model=List[ThreadSummary])
def list_threads(request: Request, response: Response):
    """Lists all conversation threads owned by the current user."""
    user_id = get_current_user(request, response)
    with logfire.span("📋 List User Threads", user_id=user_id):
        threads = get_user_threads(user_id)
        return [ThreadSummary(**t) for t in threads]


@router.get("/threads/{thread_id}", response_model=ThreadDetail)
def get_thread(thread_id: str, request: Request, response: Response):
    """Retrieves full conversation history for a specific thread owned by user."""
    user_id = get_current_user(request, response)
    with logfire.span("🔍 Get Thread Detail", thread_id=thread_id, user_id=user_id):
        try:
            detail = get_thread_detail(thread_id, user_id)
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: you do not own this thread."
            )

        if not detail:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread '{thread_id}' not found."
            )
        return ThreadDetail(**detail)


@router.delete("/threads/{thread_id}")
def delete_thread(thread_id: str, request: Request, response: Response):
    """Deletes a conversation thread and its messages if owned by the current user."""
    user_id = get_current_user(request, response)
    with logfire.span("🗑️ Delete Thread", thread_id=thread_id, user_id=user_id):
        try:
            deleted = delete_user_thread(thread_id, user_id)
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: you do not own this thread."
            )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread '{thread_id}' not found."
            )

        # Also purge checkpointer state
        from app.agents.persistence import delete_thread_checkpoint
        delete_thread_checkpoint(thread_id)

        return {"status": "ok", "message": f"Thread '{thread_id}' deleted."}
