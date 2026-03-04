"""
API endpoints for presentation slide figures.
GET /api/figures/slide1, slide2, slide3 → generate figures + return stats.
Static images: /api/static/slide1_distribution.png, etc.
"""

from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/figures", tags=["figures"])

PROJECT_DIR = Path(__file__).resolve().parents[2]
IMG_DIR = PROJECT_DIR / "report_images"


@router.get("/slide1")
def generate_slide1():
    """Generate Slide 1 (Stylized Facts): distribution, QQ, volatility clustering."""
    try:
        from src.presentation.slide1_risk_profile import generate_slide1 as _gen
        stats = _gen()
        return stats
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "error": "Failed to generate slide1"},
        )


@router.get("/slide2")
def generate_slide2():
    """Generate Slide 2 (Regime Value): CVaR, event study."""
    try:
        from src.presentation.slide2_regime_value import generate_slide2 as _gen
        stats = _gen()
        return stats
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "error": "Failed to generate slide2"},
        )


@router.get("/slide3")
def generate_slide3():
    """Generate Slide 3 (Performance & Drawdown): Sharpe, CAGR, MDD."""
    try:
        from src.presentation.slide3_performance import generate_slide3 as _gen
        stats = _gen()
        return stats
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "error": "Failed to generate slide3"},
        )


@router.get("/slide1/images")
def get_slide1_images():
    """Generate slide1 and return image paths. Frontend can then fetch /api/static/<name>."""
    try:
        from src.presentation.slide1_risk_profile import generate_slide1 as _gen
        stats = _gen()
        return {
            **stats,
            "images": [
                "/api/static/slide1_distribution.png",
                "/api/static/slide1_volatility.png",
            ],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
