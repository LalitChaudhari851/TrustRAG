import logfire
from fastapi import APIRouter, Depends, Request, Response
from app.models.api_schemas import FeedbackRequest, FeedbackResponse
from app.models.feedback_db import save_feedback
from app.api.auth import get_current_user

router = APIRouter(prefix="/v1", tags=["Feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest, request: Request, response: Response):
    """
    Submits user feedback (thumbs up / thumbs down with optional comment) for a specific message.
    Stored in a standalone SQLite database.
    """
    user_id = get_current_user(request, response)
    with logfire.span("🗳️ User Feedback Received", message_id=payload.message_id, rating=payload.rating, user_id=user_id):
        fb_id = save_feedback(
            message_id=payload.message_id,
            user_id=user_id,
            rating=payload.rating,
            comment=payload.comment
        )
        return FeedbackResponse(status="ok", feedback_id=fb_id)
