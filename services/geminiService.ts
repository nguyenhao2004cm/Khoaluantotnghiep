
import { GoogleGenAI } from "@google/genai";
import { ChatContext } from "../types";

export class GeminiService {
  private ai: any;

  constructor() {
    // Always use {apiKey: process.env.API_KEY} for initialization as per guidelines
    this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  }

  async getAdvice(message: string, context: ChatContext, history: any[] = []) {
    try {
      const systemInstruction = `
        Bạn là một chuyên gia tư vấn đầu tư tài chính cao cấp cho hệ thống "AI-driven Regime-Aware Portfolio Optimization Platform".
        Mục tiêu của bạn là giải thích kết quả danh mục dựa trên rủi ro thị trường (Regime-Aware).
        
        Bối cảnh hiện tại của người dùng:
        - Trạng thái thị trường: ${context.market_regime}
        - Danh mục tài sản: ${context.user_assets.join(', ')}
        - Chỉ số hiệu suất: CAGR=${(context.portfolio_metrics.CAGR * 100).toFixed(2)}%, Max Drawdown=${(context.portfolio_metrics.maxDrawdown * 100).toFixed(2)}%, CVaR_5=${(context.portfolio_metrics.CVaR_5 * 100).toFixed(2)}%
        - Chiến lược đang áp dụng: ${context.current_strategy}

        Quy tắc ứng xử:
        1. GIẢI THÍCH vì sao AI chọn phân bổ này dựa trên trạng thái thị trường ${context.market_regime}.
        2. TRẢ LỜI các câu hỏi "Nếu như" (e.g. Nếu thị trường chuyển sang LOW rủi ro?).
        3. KHÔNG dự đoán giá chứng khoán cụ thể.
        4. KHÔNG gợi ý mua/bán một mã cụ thể ngoài việc giải thích logic phân bổ chung.
        5. KHÔNG thay đổi quyết định của bộ máy tối ưu hóa (optimizer).
        6. Luôn sử dụng tiếng Việt chuyên nghiệp, hiện đại, dễ hiểu.
      `;

      // Use ai.models.generateContent to query GenAI with model and parameters
      const response = await this.ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: [
            ...history,
            { role: 'user', parts: [{ text: message }] }
        ],
        config: {
          systemInstruction,
          temperature: 0.7,
        }
      });

      // Extract generated text using the .text property
      return response.text;
    } catch (error) {
      console.error("Gemini API Error:", error);
      return "Xin lỗi, tôi gặp trục trặc khi kết nối với máy chủ AI. Vui lòng thử lại sau.";
    }
  }
}

export const geminiService = new GeminiService();
