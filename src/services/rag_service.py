import time
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from src.config.config_parser import settings
from src.logging.logger import logger
from src.vectors.database import VectorDatabase
from src.core.factories import ModelFactory
from src.vectors.database import VectorDatabase


class RAGService:
    def __init__(self):
        self.repo = VectorDatabase()
        self.llm = ModelFactory.get_llm()
        self.template = """
أنت موظف خدمة عملاء في مزود خدمة إنترنت (ISP).
مهمتك هي الرد على شكوى العميل بالعامية المصرية بطريقة مهذبة واحترافية.
ممنوع تمامًا:
- ذكر أي اسم شركة اتصالات حقيقي
- ذكر أنك ذكاء اصطناعي أو بوت
- إعطاء رقم/إيميل/رابط غير موجود حرفيًا في السياق
- اختراع معلومة غير موجودة في السياق؛ حوّلها لفريق مختص لو ناقصة
- استخدام ألفاظ رسمية زي "سيادتكم" أو "حضرتكم الموقر"
- تكرار نفس جملة الافتتاح في كل رد
اذكر الحل في نقط مرقمة لو فيه أكتر من خطوة، وابدأ دايمًا بذكر كود الخطأ لو العميل ذكره.
السياق الداخلي:
{context}
شكوى العميل:
{question}
الرد:
"""
        self.prompt = PromptTemplate(template=self.template)

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer_ticket(self, customer_ticket: str) -> dict:
        start_time = time.time()
        logger.info(f"Processing customer ticket query: {customer_ticket[:50]}...")

        # Load Vector Store
        vectorstore = self.repo.load_index()
        retriever = vectorstore.as_retriever(search_kwargs={"k": settings.k_retrieval})

        # Fetch chunks for audit/telemetry
        retrieved_chunks = retriever.invoke(customer_ticket)
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks from FAISS.")

        # Build LCEL Chain (up to LLM to return AIMessage with metadata)
        rag_chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
        )

        # Generate Response AIMessage
        ai_message = rag_chain.invoke(customer_ticket)
        content = getattr(ai_message, "content", "")

        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            response_text = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            response_text = str(content)

        # Extract Token Usage Telemetry
        usage = getattr(ai_message, "usage_metadata", None) or {}
        
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Fallback check for response_metadata
        if not prompt_tokens and hasattr(ai_message, "response_metadata"):
            meta = ai_message.response_metadata.get("token_usage", {})
            prompt_tokens = meta.get("prompt_tokens") or meta.get("input_tokens") or 0
            completion_tokens = (
                meta.get("completion_tokens") or meta.get("output_tokens") or 0
            )
            total_tokens = meta.get("total_tokens") or (
                prompt_tokens + completion_tokens
            )

        elapsed_time = round(time.time() - start_time, 2)
        logger.info(
            f"RAG answer generated in {elapsed_time}s. "
            f"Tokens: Input={prompt_tokens}, Output={completion_tokens}, Total={total_tokens}"
        )

        return {
            "ticket": customer_ticket,
            "response": response_text,
            "sources_count": len(retrieved_chunks),
            "execution_time_seconds": elapsed_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
