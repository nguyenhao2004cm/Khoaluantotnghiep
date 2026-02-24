# POST /api/advisor/chat
# Nhận message + context + history → gọi Gemini → trả { answer }

import os
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

from .schemas import ChatRequest, ChatResponse

load_dotenv()

router = APIRouter(prefix="/api/advisor", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def post_advisor_chat(body: ChatRequest):
    """
    AI Chat: nhận câu hỏi + context (regime, metrics, strategy) → Gemini → answer.
    API key Gemini chỉ ở backend.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return ChatResponse(
            answer="Xin lỗi, dịch vụ AI chưa được cấu hình (thiếu API key). Vui lòng liên hệ quản trị."
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        ctx = body.context

        system_instruction = (
            "Bạn là chuyên gia tư vấn phân tích danh mục đầu tư cho hệ thống Regime-Aware Portfolio Optimization. "
            "Nhiệm vụ: giải thích kết quả danh mục và rủi ro theo regime; KHÔNG dự đoán giá; KHÔNG khuyến nghị mua/bán cụ thể. "
            f"Bối cảnh: Regime={ctx.market_regime}, Chiến lược={ctx.current_strategy}, "
            f"Tài sản={ctx.user_assets}, Metrics: CAGR={ctx.portfolio_metrics.cagr:.2%}, MaxDD={ctx.portfolio_metrics.max_drawdown:.2%}, CVaR_5={ctx.portfolio_metrics.cvar_5:.2%}. "
            "Trả lời ngắn gọn, học thuật, bằng tiếng Việt."
        )

        contents = []
        for msg in body.history[-10:]:  # last 10 turns
            role = "user" if msg.role == "user" else "model"
            text = msg.parts[0].text if msg.parts else ""
            contents.append({"role": role, "parts": [{"text": text}]})
        contents.append({"role": "user", "parts": [{"text": body.message}]})

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            ),
        )
        answer = (response.text or "").strip() or "Không có phản hồi."
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advisor error: {str(e)}")
