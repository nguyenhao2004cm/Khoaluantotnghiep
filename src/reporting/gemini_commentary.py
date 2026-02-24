import json
import time
import pandas as pd
from google import genai

MODEL = "gemini-2.5-flash"

def generate_commentary_once(
    perf_dict: dict,
    risk_dict: dict,
    frontier_dict: dict,
    corr_summary: dict,
    annual_returns_df: pd.DataFrame,
    client
) -> dict:
    """
    Gọi Gemini DUY NHẤT 1 LẦN.
    Trả về dict JSON cho toàn bộ báo cáo.
    """

    annual_text = annual_returns_df.head(5).to_string(index=False)

    prompt = f"""
Bạn là chuyên gia phân tích danh mục đầu tư và quản trị rủi ro định lượng.
Dựa trên các dữ liệu định lượng của danh mục đầu tư tối ưu được cung cấp dưới đây,
hãy viết các nhận xét học thuật, trung tính, phục vụ cho báo cáo nghiên cứu tài chính.

DỮ LIỆU HIỆU SUẤT:
{json.dumps(perf_dict, ensure_ascii=False)}

DỮ LIỆU RỦI RO:
{json.dumps(risk_dict, ensure_ascii=False)}

LỢI NHUẬN THEO NĂM:
{annual_text}

ĐƯỜNG BIÊN HIỆU QUẢ:
{json.dumps(frontier_dict, ensure_ascii=False)}

TƯƠNG QUAN:
{json.dumps(corr_summary, ensure_ascii=False)}
========================
YÊU CẦU VIẾT NHẬN XÉT
========================

Bạn phải tuân thủ NGHIÊM NGẶT các quy định sau:

1. Mỗi khóa trong JSON chỉ được nhận xét dựa trên ĐÚNG NHÓM THÔNG TIN liên quan.
   Không được lặp ý, không được dùng thông tin của mục khác.

2. Quy định nội dung cho từng mục:

Mỗi mục nhận xét PHẢI gắn trực tiếp với BIỂU ĐỒ hoặc BẢNG tương ứng trong báo cáo.
Không được viết chung chung, không được tái sử dụng luận điểm của mục khác.

Quy định bắt buộc:

- performance:
  Nhận xét DUY NHẤT dựa trên biểu đồ tăng trưởng giá trị danh mục theo thời gian.
  Phải đề cập đến xu hướng dài hạn, các pha tăng – điều chỉnh – phục hồi.
  Không nhắc đến chỉ số Sharpe, VaR, CVaR hay tương quan.

- portfolio_overview:
  Nhận xét DUY NHẤT dựa trên bảng tổng quan chỉ số.
  Phải đề cập đến mối quan hệ giữa CAGR – Volatility – Drawdown – Tail risk.
  Không nhắc đến hình dạng phân phối hoặc diễn biến theo thời gian.

- annual_returns:
  Nhận xét DUY NHẤT dựa trên bảng lợi nhuận theo năm.
  Phải đề cập đến tính chu kỳ, sự phân kỳ giữa các năm, và tính ổn định liên thời gian.
  Không nhắc đến biểu đồ tăng trưởng tổng.

- risk:
  Nhận xét DUY NHẤT dựa trên biểu đồ drawdown và các chỉ số rủi ro.
  Phải đề cập đến độ sâu, thời gian phục hồi, và rủi ro đuôi.
  Không nhắc đến lợi nhuận kỳ vọng.

- frontier:
  Giải thích vị trí danh mục AI so với đường biên hiệu quả CHỈ MANG TÍNH THAM KHẢO.
  Phải nêu rõ đây không phải danh mục Markowitz tối ưu.
  Không được kết luận danh mục là “tối ưu theo Markowitz”.

- correlation:
  Nhận xét DUY NHẤT dựa trên chỉ số tương quan tổng hợp.
  Phải liên hệ đến khả năng đa dạng hóa và giảm rủi ro hệ thống.

- recommendation (Khuyến nghị đầu tư):
  BẮT BUỘC: Nhận xét TỔNG HỢP về toàn bộ danh mục dựa trên tất cả dữ liệu đã cung cấp.
  Phải tóm tắt điểm mạnh, điểm yếu, mức độ rủi ro và triển vọng của danh mục.
  CẤM: Không được yêu cầu phân tích thêm, không được khuyến nghị "tiếp tục phân tích chi tiết",
  không được yêu cầu cung cấp thêm thông tin. Phần này phải là KẾT LUẬN TỔNG HỢP, không phải gợi ý hành động tiếp theo.

========================
TRẢ KẾT QUẢ DƯỚI DẠNG JSON
========================


{{
  "performance": "...",
  "risk": "...",
  "annual_returns": "...",
  "frontier": "...",
  "correlation": "...",
  "portfolio_overview": "...",
  "recommendation": "..."
}}
"""

    time.sleep(2)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip()

    # Bóc JSON an toàn
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        out = json.loads(text[json_start:json_end])
        # Đảm bảo có đủ keys (Gemini đôi khi bỏ sót)
        defaults = {
            "performance": "Nhận xét hiệu suất danh mục.",
            "risk": "Nhận xét rủi ro danh mục.",
            "annual_returns": "Nhận xét lợi nhuận theo năm.",
            "frontier": "Vị trí danh mục so với đường biên hiệu quả mang tính tham khảo.",
            "correlation": "Nhận xét tương quan và đa dạng hóa.",
            "portfolio_overview": "Tổng quan chỉ số danh mục.",
            "recommendation": "Nhận xét tổng hợp về danh mục dựa trên hiệu suất, rủi ro và đa dạng hóa.",
        }
        for k, v in defaults.items():
            if k not in out or not out[k]:
                out[k] = v
        return out
    except Exception:
        raise ValueError("❌ Gemini không trả JSON hợp lệ")
