# FastAPI app: mount API layer, CORS cho frontend

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .market import router as market_router
from .portfolio import router as portfolio_router
from .report import router as report_router
from .chat import router as chat_router
from .company import router as company_router

app = FastAPI(
    title="AI Regime-Aware Portfolio API",
    description="Backend API cho web frontend: market overview, portfolio optimize, report PDF, AI chat.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "https://khoaluantotnghiep-jet.vercel.app",
        "https://www.khoaluantotnghiep-jet.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router)
app.include_router(portfolio_router)
app.include_router(report_router)
app.include_router(chat_router)
app.include_router(company_router)


@app.get("/")
def root():
    return {"message": "AI Regime-Aware Portfolio API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
