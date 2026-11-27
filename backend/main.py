"""CiteBase 文档问答的 FastAPI 入口。"""

from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints
from requests import RequestException, Timeout

from client import User
from router import DocumentRouter


class SessionInput(BaseModel):
    role: str = ""
    context: str = ""


class SessionInfo(BaseModel):
    session_id: UUID


class QuestionInput(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConversationTurn(BaseModel):
    question: str
    answer: str


class SessionHistory(SessionInfo):
    history: list[ConversationTurn]


@dataclass
class Session:
    user: User
    lock: Lock = field(default_factory=Lock)


app = FastAPI(title="CiteBase")
sessions: dict[UUID, Session] = {}


@lru_cache(maxsize=1)
def get_router() -> DocumentRouter:
    """首次创建会话时载入默认文档，供后续会话共享。"""
    return DocumentRouter()


def get_session(session_id: UUID) -> Session:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@app.post("/sessions", status_code=201)
def create_session(payload: SessionInput) -> SessionInfo:
    """创建独立会话，可设置回答角色和背景。"""
    try:
        router = get_router()
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail="无法初始化文档问答") from error
    session_id = uuid4()
    sessions[session_id] = Session(User(router, role=payload.role, context=payload.context))
    return SessionInfo(session_id=session_id)


@app.post("/sessions/{session_id}/questions")
def ask_question(session_id: UUID, payload: QuestionInput) -> ConversationTurn:
    """提交问题，按会话顺序生成回答并记录本轮问答。"""
    session = get_session(session_id)
    with session.lock:
        try:
            answer = session.user.ask(payload.question)
        except Timeout as error:
            raise HTTPException(status_code=504, detail="上游问答请求超时") from error
        except (RequestException, ValueError) as error:
            raise HTTPException(status_code=502, detail="上游问答请求失败") from error
    return ConversationTurn(question=payload.question, answer=answer)


@app.get("/sessions/{session_id}/history")
def read_history(session_id: UUID) -> SessionHistory:
    """按提问顺序返回会话已完成的问答。"""
    session = get_session(session_id)
    with session.lock:
        history = session.user.history
    return SessionHistory(
        session_id=session_id,
        history=[ConversationTurn(question=question, answer=answer) for question, answer in history],
    )
