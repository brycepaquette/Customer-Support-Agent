from typing import Any

from langfuse import observe
from pydantic import BaseModel, Field


class EchoArgs(BaseModel):
    text: str = Field(..., description="Text to echo back")


class EchoResult(BaseModel):
    text: str
    upper: str


@observe()
def echo(args: EchoArgs) -> EchoResult:
    """Return the input text, plus its uppercase form."""
    return EchoResult(text=args.text, upper=args.text.upper())


ECHO_TOOL: dict[str, Any] = {
    "name": "echo",
    "description": "Echo the provided text and return it alongside its uppercase form.",
    "input_schema": EchoArgs.model_json_schema(),
}
