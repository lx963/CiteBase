import os
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_setting(name: str) -> str:
    """优先读取环境变量，其次读取项目 .env 中的 NAME=value 配置。"""
    value = os.getenv(name, "").strip()
    if value:
        return value

    env_path = PROJECT_ROOT / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return ""

    for line in lines:
        variable, separator, value = line.partition("=")
        if separator and variable.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value.strip()
    return ""


read_api_key = read_setting


def format_api_error(error: requests.HTTPError, *, service: str = "API") -> str:
    """提取 HTTP 状态和服务返回的错误消息。"""
    response = error.response
    if response is None:
        return f"{service} 请求失败：{error}"

    try:
        body = response.json()
    except requests.exceptions.JSONDecodeError:
        body = {}
    details = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(details, dict):
        details = {}

    code = details.get("code")
    status = f"HTTP {response.status_code}"
    if code is not None and code != response.status_code:
        status += f" / {code}"
    message = details.get("message") or response.reason
    return f"{service} 请求失败（{status}）{message}"


def format_openrouter_error(error: requests.HTTPError) -> str:
    """提取 OpenRouter 的 HTTP 状态和错误消息。"""
    return format_api_error(error, service="OpenRouter")
