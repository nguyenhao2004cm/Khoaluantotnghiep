# GET /api/report/download
# Tra file PDF tu ket qua optimize gan nhat hoac report mac dinh

from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from .portfolio import get_last_optimization_result

router = APIRouter(prefix="/api/report", tags=["report"])

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_DIR / "Reports"


def _serve_latest_pdf():
    if REPORTS_DIR.exists():
        pdfs = list(REPORTS_DIR.glob("*.pdf"))
        if pdfs:
            latest = max(pdfs, key=lambda p: p.stat().st_mtime)
            return FileResponse(
                path=str(latest),
                media_type="application/pdf",
                filename="Portfolio_Optimization_Report.pdf",
                headers={"Content-Disposition": "attachment; filename=\"Portfolio_Optimization_Report.pdf\""},
            )
    return JSONResponse(
        status_code=404,
        content={"detail": "Chưa có báo cáo PDF. Vui lòng chạy tối ưu danh mục thành công trước (hoặc chạy python run_all.py)."},
    )


@router.get("/download")
def get_report_download():
    return _serve_latest_pdf()


@router.get("/pdf")
def get_report_pdf():
    return _serve_latest_pdf()
