from typing import Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    message: str = Field(..., description="Ingest response message")
    chunks_indexed: int = Field(..., description="Number of chunks indexed")
    index_path: str = Field(..., description="Path to the FAISS index file")


class QueryRequest(BaseModel):
    ticket: str = Field(
        ...,
        description="Customer problem description in Arabic/English",
        example="النت فاصل عندي ولمبة DSL بتنور وتطفي",
    )


class QueryResponse(BaseModel):
    ticket: str = Field(..., description="Original customer query")
    response: str = Field(..., description="Generated Arabic AI response")

    candidates_count: int = Field(
        ...,
        ge=0,
        description="Number of chunks retrieved as candidates",
    )

    sources_count: int = Field(
        ...,
        ge=0,
        description="Number of chunks actually included in the final LLM context",
    )

    execution_time_seconds: float = Field(
        ...,
        description="Latency in seconds",
    )

    prompt_tokens: int = Field(
        0,
        description="Actual input tokens consumed by the LLM",
    )

    completion_tokens: int = Field(
        0,
        description="Actual output tokens consumed by the LLM",
    )

    total_tokens: int = Field(
        0,
        description="Total tokens consumed",
    )

    input_cost_usd: float = Field(
        0,
        description="Estimated/actual input cost in USD",
    )

    output_cost_usd: float = Field(
        0,
        description="Estimated/actual output cost in USD",
    )

    total_cost_usd: float = Field(
        0,
        description="Estimated/actual total cost in USD",
    )


class IngestRequest(BaseModel):
    file_path: Optional[str] = Field(
        None,
        description="Custom document path (optional)",
    )