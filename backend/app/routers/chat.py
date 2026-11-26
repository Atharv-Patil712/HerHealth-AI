"""Server-side proxy for the Wellspring health chat."""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter(tags=["chat"])

SYSTEM_PROMPT = """You are Wellspring, a warm, knowledgeable women's health companion focused on reproductive and general women's health.

Give clear, accurate, general health education in plain, everyday language. You are not a doctor and cannot diagnose anyone. Explain common associations with symptoms, and encourage appropriate evaluation by a gynecologist or GP.

Do not give specific medications or dosages. If symptoms could be urgent, calmly tell the person to seek urgent or emergency care now. Keep answers focused and conversational, and never present general information as a diagnosis."""


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=30)


class ChatResponse(BaseModel):
    content: str


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="Chat is unavailable because the Anthropic API key is not configured.",
        )

    payload = {
        "model": settings.anthropic_model,
        "max_tokens": 1000,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="The chat provider could not respond.") from exc

    data = response.json()
    text = next(
        (block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"),
        "",
    )
    if not text:
        raise HTTPException(status_code=502, detail="The chat provider returned an empty response.")
    return ChatResponse(content=text)
