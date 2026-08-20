import time
from langchain_core.prompts import PromptTemplate
from src.config.config_parser import settings
from src.services.cost_calculator import CostCalculator
from src.core.factories import ModelFactory
from src.logging.logger import logger
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

اذكر الحل في نقط مرقمة لو فيه أكتر من خطوة،
وابدأ دايمًا بذكر كود الخطأ لو العميل ذكره.

- خلي الرد مختصر وواضح ومباشر.
- لا تتجاوز 5 خطوات في الحل.
- لا تكرر المعلومات الموجودة في الشكوى.
- لا تضف مقدمات أو خاتمات طويلة.
- أنهِ الرد فور اكتمال الحل.

السياق الداخلي:
{context}

شكوى العميل:
{question}

الرد:
"""

        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"],
        )

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_prompt(
        self,
        context: str,
        question: str,
    ) -> str:
        return self.prompt.format(
            context=context,
            question=question,
        )

    def _select_context_chunks(
        self,
        customer_ticket: str,
        candidate_chunks: list,
    ) -> list:
        selected_chunks = []

        for chunk in candidate_chunks:
            trial_chunks = selected_chunks + [chunk]

            context = self._format_docs(trial_chunks)

            final_prompt = self._build_prompt(
                context=context,
                question=customer_ticket,
            )

            token_count = self.llm.get_num_tokens(final_prompt)

            if token_count <= settings.max_input_tokens:
                selected_chunks.append(chunk)
            else:
                break

        return selected_chunks

    def answer_ticket(self, customer_ticket: str) -> dict:
        start_time = time.time()

        logger.info(f"Processing customer ticket query: " f"{customer_ticket[:50]}...")

        # Load vector store
        vectorstore = self.repo.load_index()

        # Retrieve candidate chunks ONCE
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": settings.k_retrieval,
            }
        )

        candidate_chunks = retriever.invoke(customer_ticket)

        logger.info(
            f"Retrieved {len(candidate_chunks)} " f"candidate chunks from FAISS."
        )

        # Select only chunks that fit inside input token budget
        selected_chunks = self._select_context_chunks(
            customer_ticket=customer_ticket,
            candidate_chunks=candidate_chunks,
        )

        logger.info(
            f"Selected {len(selected_chunks)} chunks " f"for final LLM context."
        )

        context = self._format_docs(selected_chunks)

        final_prompt = self._build_prompt(
            context=context,
            question=customer_ticket,
        )

        estimated_input_tokens = self.llm.get_num_tokens(final_prompt)

        logger.info(
            f"Estimated input tokens before LLM call: " f"{estimated_input_tokens}"
        )

        # Call Gemini exactly once
        ai_message = self.llm.invoke(final_prompt)
        logger.info(f"Response metadata: {ai_message.response_metadata}")
        
        content = getattr(
            ai_message,
            "content",
            "",
        )

        if isinstance(content, str):
            response_text = content

        elif isinstance(content, list):
            response_text = "\n".join(
                block.get("text", "")
                for block in content
                if (isinstance(block, dict) and block.get("type") == "text")
            )

        else:
            response_text = str(content)

        # Extract real usage from Gemini response
        usage = (
            getattr(
                ai_message,
                "usage_metadata",
                None,
            )
            or {}
        )

        prompt_tokens = usage.get(
            "input_tokens",
            0,
        )

        completion_tokens = usage.get(
            "output_tokens",
            0,
        )

        total_tokens = usage.get(
            "total_tokens",
            prompt_tokens + completion_tokens,
        )

        # Backward compatibility fallback
        if not prompt_tokens and hasattr(
            ai_message,
            "response_metadata",
        ):
            metadata = ai_message.response_metadata or {}

            token_usage = metadata.get(
                "token_usage",
                {},
            )

            prompt_tokens = (
                token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
            )

            completion_tokens = (
                token_usage.get("completion_tokens")
                or token_usage.get("output_tokens")
                or 0
            )

            total_tokens = (
                token_usage.get("total_tokens") or prompt_tokens + completion_tokens
            )

        # Calculate actual cost
        cost = CostCalculator.calculate(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )

        elapsed_time = round(
            time.time() - start_time,
            2,
        )

        logger.info(
            f"RAG answer generated in {elapsed_time}s. "
            f"Tokens: "
            f"Input={prompt_tokens}, "
            f"Output={completion_tokens}, "
            f"Total={total_tokens}. "
            f"Cost: "
            f"${cost['total_cost_usd']:.8f}"
        )

        return {
            "ticket": customer_ticket,
            "response": response_text,
            "candidates_count": len(candidate_chunks),
            "sources_count": len(selected_chunks),
            "execution_time_seconds": elapsed_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_cost_usd": cost["input_cost_usd"],
            "output_cost_usd": cost["output_cost_usd"],
            "total_cost_usd": cost["total_cost_usd"],
        }
