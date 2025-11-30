from pydantic import BaseModel
from typing import Literal, Dict, Any


class StatusMessage(BaseModel):
    type: Literal["tool_start", "tool_end"]
    name: str
    arguments: Dict[str, Any]
