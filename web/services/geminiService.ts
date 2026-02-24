// Trang web/services/geminiService.ts
// Explainability layer: frontend chỉ gọi backend, không gọi Gemini trực tiếp.

import { ChatContext, ChatMessage } from "../types";
import { API_BASE } from "../apiConfig";

export class GeminiService {
  async getAdvice(
    message: string,
    context: ChatContext,
    history: ChatMessage[] = []
  ): Promise<string> {
    try {
      const res = await fetch(`${API_BASE}/api/advisor/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          context,
          history,
        }),
      });

      if (!res.ok) {
        throw new Error("Advisor API failed");
      }

      const data = await res.json();
      return data.answer;
    } catch (error) {
      console.error("Advisor Service Error:", error);
      return "Xin lỗi, tôi không thể truy xuất phân tích AI vào lúc này.";
    }
  }
}

export const geminiService = new GeminiService();

/** Convenience wrapper for AI Advisor */
export async function askAdvisor(
  message: string,
  context: ChatContext,
  history: ChatMessage[] = []
): Promise<string> {
  return geminiService.getAdvice(message, context, history);
}
