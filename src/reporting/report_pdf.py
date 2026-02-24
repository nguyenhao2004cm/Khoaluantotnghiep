##C:\Users\ASUS\fintech-project\src\reporting\report_pdf.py
# ======================================================
# PORTFOLIO OPTIMIZATION REPORT (FINAL CLEAN VERSION)
# ======================================================

import os
import sys
from pathlib import Path

# Cho phép chạy trực tiếp: python report_pdf.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, PageBreak
)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from matplotlib.lines import Line2D
from src.reporting.gemini_commentary import generate_commentary_once
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from src.portfolio_engine.performance_metrics_genai import compute_performance_metrics
from src.portfolio_engine.risk_metric import compute_risk_metrics
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import math
from google import genai
import json
from dotenv import load_dotenv
load_dotenv()

# ======================================================
# CONFIG
# ======================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_POWERBI_DIR = PROJECT_DIR / "data_processed" / "powerbi"
DATA_REPORT_DIR  = PROJECT_DIR / "data_processed" / "reporting"

IMG_DIR = PROJECT_DIR / "report_images"
IMG_DIR.mkdir(exist_ok=True)

OUT_DIR = PROJECT_DIR / "Reports"
OUT_DIR.mkdir(exist_ok=True)

REPORT_DATE = datetime.now().strftime("%Y%m%d")
OUT_PDF = OUT_DIR / f"Portfolio_Optimization_Report_{REPORT_DATE}.pdf"

# ======================================================
# INVESTMENT WINDOW (CRITICAL)
# ======================================================
INVESTMENT_START_DATE = "2020-01-01"  # hoặc lấy từ user input

# ======================================================
# FONT SETUP (GLOBAL)
# ======================================================

pdfmetrics.registerFont(TTFont("Times", r"C:\Windows\Fonts\times.ttf"))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 120


def save_fig(name: str):
    path = IMG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ======================================================
# CHARTS
# ======================================================

def plot_portfolio_growth():
    df = pd.read_csv(
    DATA_POWERBI_DIR / "portfolio_timeseries.csv",
    parse_dates=["date"]
)

    df = df[df["date"] >= INVESTMENT_START_DATE]


    plt.figure(figsize=(10, 4))

    # ===== LINE – PORTFOLIO VISUALIZER BLUE =====
    plt.plot(
        df["date"],
        df["portfolio_value"],
        color="#3b6edc",
        linewidth=2.4
    )

    # ===== TITLE (KHÔNG BOLD – GIỐNG PV) =====
    plt.title(
        "Tăng trưởng giá trị danh mục (VND)",
        fontsize=13,
        fontweight="normal",
        pad=10,
        color="#1f2933"
    )

    plt.xlabel("Thời gian", fontsize=11)
    plt.ylabel("Giá trị danh mục (VND)", fontsize=11)

    # ===== GRID NHẸ, THANH =====
    plt.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.35
    )

    plt.tight_layout()
    return save_fig("portfolio_growth.png")


def plot_drawdown():
    df = pd.read_csv(
        DATA_POWERBI_DIR / "portfolio_timeseries.csv",
        parse_dates=["date"]
    )

    df = df[df["date"] >= INVESTMENT_START_DATE]
    drawdown_pct = df["drawdown"] * 100


    plt.figure(figsize=(12, 4))

    # ===== FILL – PORTFOLIO VISUALIZER BLUE =====
    plt.fill_between(
        df["date"],
        drawdown_pct,
        0,
        color="#3b6edc",
        alpha=0.28,
        linewidth=0
    )

    # ===== TITLE =====
    plt.title(
        "Mức sụt giảm danh mục (Drawdown)",
        fontsize=13,
        fontweight="normal",
        pad=10,
        color="#1f2933"
    )

    plt.xlabel("Thời gian", fontsize=11)
    plt.ylabel("Sụt giảm (%)", fontsize=11)

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.1f}%")
    )

    # ===== GRID CHỈ TRỤC Y – NHẸ =====
    plt.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.35
    )

    plt.tight_layout()
    return save_fig("drawdown.png")


def _plot_asset_allocation_chart():
    df = pd.read_csv(DATA_POWERBI_DIR / "asset_allocation_current.csv")

    # Sort theo tỷ trọng giảm dần
    df = df.sort_values("allocation_weight", ascending=False)

    plt.figure(figsize=(5.6, 5.6))

    # ===== PALETTE PORTFOLIO VISUALIZER – #3b6edc =====
    palette = [
        "#dbe7fb",
        "#a9c4f5",
        "#7fa3ee",
        "#4f7fe3",
        "#3b6edc",
    ]

    colors = palette[:len(df)]

    wedges, texts, autotexts = plt.pie(
        df["allocation_weight"],
        startangle=90,
        counterclock=False,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
        pctdistance=0.78,
        textprops={
            "fontsize": 9,
            "color": "#1f2933",
            "fontweight": "normal"
        },
        wedgeprops={
            "width": 0.38,
            "edgecolor": "white",
            "linewidth": 1.1
        }
    )

    plt.title(
        "Phân bổ tài sản danh mục",
        fontsize=13,
        fontweight="normal",
        pad=10
    )

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            label=label,
            markerfacecolor=colors[i % len(colors)],
            markeredgecolor="none",
            markersize=8
        )
        for i, label in enumerate(df["symbol"])
    ]

    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
        fontsize=9
    )

    plt.tight_layout()
    return save_fig("asset_allocation.png")


def plot_asset_allocation():
    # Wrapper để tránh UnboundLocalError
    return _plot_asset_allocation_chart()


def plot_annual_returns():
    df = pd.read_csv(DATA_POWERBI_DIR / "annual_returns.csv")

    plt.figure(figsize=(5.4, 4))

    plt.bar(
        df["year"],
        df["annual_return"] * 100,
        color="#3b6edc",
        width=0.6
    )

    plt.title(
        "Lợi nhuận danh mục theo năm",
        fontsize=13,
        fontweight="normal",
        pad=10,
        color="#1f2933"
    )
    plt.xlabel("Năm", fontsize=11)
    plt.ylabel("Lợi nhuận (%)", fontsize=11)

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.1f}%")
    )

    plt.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    plt.tight_layout()
    return save_fig("annual_returns.png")


def plot_risk_return():
    df = pd.read_csv(DATA_POWERBI_DIR / "asset_risk_return.csv")

    plt.figure(figsize=(5.8, 4.4))

    plt.scatter(
        df["volatility"],
        df["expected_return"],
        s=75,
        color="#3b6edc",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8
    )

    for _, r in df.iterrows():
        plt.text(
            r["volatility"],
            r["expected_return"] + 0.002,
            r["symbol"],
            fontsize=9,
            ha="center",
            va="bottom",
            color="#1f2933",
            alpha=0.8
        )

    # ===== TREND LINE =====
    x = df["volatility"].values
    y = df["expected_return"].values
    coef = np.polyfit(x, y, 1)
    trend = np.poly1d(coef)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = trend(x_line)

    plt.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=1.6,
        color="#a9c4f5",
        label="Xu hướng rủi ro – lợi nhuận"
    )

    plt.title(
        "Rủi ro – Lợi nhuận",
        fontsize=13,
        fontweight="normal",
        pad=10,
        color="#1f2933"
    )
    plt.xlabel("Rủi ro (Độ biến động)", fontsize=11)
    plt.ylabel("Lợi nhuận kỳ vọng", fontsize=11)

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%")
    )

    plt.legend(frameon=False, fontsize=9)
    plt.grid(linestyle="--", linewidth=0.8, alpha=0.35)
    plt.tight_layout()
    return save_fig("risk_return.png")

from src.portfolio_engine.efficient_frontier import (
    plot_efficient_frontier_reference
)

# Lấy weights hiện tại (1 ngày, hoặc ngày cuối)
weights = (
    pd.read_csv("data_processed/portfolio/portfolio_allocation_final.csv")
      .groupby("symbol")["allocation_weight"]
      .last()
      .to_dict()
)

symbols = list(weights.keys())



def plot_risk_return_metrics():
    df = pd.read_csv(DATA_REPORT_DIR / "performance_extended.csv")

    df.columns = df.columns.str.lower()
    df["metric"] = df["metric"].str.strip().str.lower()

    # =========================
    # METRICS GỐC (KEY NỘI BỘ)
    # =========================
    percent_metrics = {
        "volatility_annual": "Độ biến động hàng năm",
        "max_drawdown": "Mức sụt giảm tối đa",
        "var_5": "Giá trị rủi ro lịch sử (VaR 5%)",
        "cvar_5": "Giá trị rủi ro có điều kiện (CVaR 5%)",
    }

    ratio_metrics = {
        "sharpe_ratio": "Chỉ số Sharpe",
        "sortino_ratio": "Chỉ số Sortino",
    }

    metric_map = {**percent_metrics, **ratio_metrics}

    # =========================
    # FILTER ĐÚNG METRIC
    # =========================
    df = df[df["metric"].isin(metric_map.keys())].copy()

    if df.empty:
        raise ValueError(" Không có metric hợp lệ để vẽ risk_return_metrics")

    # =========================
    # VALUE CHUẨN HÓA
    # =========================
    def parse_value(v):
        if isinstance(v, str):
            v = v.replace("%", "").strip()
        try:
            return float(v)
        except:
            return np.nan

    df["plot_value"] = df["value"].apply(parse_value)

    # =========================
    # LABEL TIẾNG VIỆT
    # =========================
    df["label_vi"] = df["metric"].map(metric_map)

    # =========================
    # VẼ BIỂU ĐỒ
    # =========================
    plt.figure(figsize=(6.4, 4.4))

    plt.barh(
        df["label_vi"],
        df["plot_value"],
        color="#3b6edc"
    )

    plt.title(
        "Chỉ số rủi ro và hiệu suất danh mục",
        fontsize=13,
        color="#1f2933"
    )

    plt.gca().invert_yaxis()  # metric quan trọng ở trên

    plt.grid(
        axis="x",
        linestyle="--",
        linewidth=0.8,
        alpha=0.35
    )

    plt.tight_layout()
    return save_fig("risk_return_metrics.png")


def plot_return_distribution():
    df = pd.read_csv(DATA_POWERBI_DIR / "portfolio_timeseries.csv")
    returns = df["portfolio_return"].dropna() * 100

    plt.figure(figsize=(6, 4))

    plt.hist(
        returns,
        bins=30,
        color="#3b6edc",
        edgecolor="white",
        alpha=0.85
    )

    plt.title(
        "Phân phối lợi nhuận danh mục",
        fontsize=13,
        fontweight="normal",
        color="#1f2933"
    )
    plt.xlabel("Lợi nhuận (%)", fontsize=11)

    plt.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    plt.tight_layout()
    return save_fig("return_distribution.png")


# =========================
# CONFIG
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT_FILE = REPORT_DIR / f"Portfolio_Optimization_Report_{datetime.now():%Y%m%d}.pdf"

DATA_DIR = PROJECT_DIR / "data_processed" / "reporting"

FILES = {
    "performance": DATA_DIR / "performance_extended.csv",
    "annual": DATA_DIR / "annual_returns_table.csv",
    "drawdown": DATA_DIR / "drawdown_periods.csv",
}


METRIC_LABEL_VI = {
    "start_balance": "Giá trị danh mục ban đầu",
    "end_balance": "Giá trị danh mục cuối kỳ",

    "cagr": "Tốc độ tăng trưởng kép hàng năm (CAGR)",
    "mean_return_annual": "Lợi nhuận trung bình năm",
    "volatility_annual": "Độ biến động hàng năm",

    "sharpe_ratio": "Chỉ số Sharpe",
    "sortino_ratio": "Chỉ số Sortino",

    "max_drawdown": "Mức sụt giảm tối đa",
    "best_day_return": "Lợi nhuận ngày cao nhất",
    "worst_day_return": "Lợi nhuận ngày thấp nhất",
    "positive_days_ratio": "Tỷ lệ ngày sinh lời",

    "var_5": "Giá trị rủi ro lịch sử (VaR 5%)",
    "cvar_5": "Giá trị rủi ro có điều kiện (CVaR 5%)",

    "skewness": "Độ lệch phân phối",
    "excess_kurtosis": "Độ nhọn phân phối",
}


# =========================
# UTILS
# =========================
def load_table(csv_file):
    df = pd.read_csv(csv_file)

    # Thay NaN / None bằng chuỗi rỗng
    df = df.where(pd.notnull(df), "")

    data = [df.columns.tolist()] + df.values.tolist()
    return data


def styled_table(data, col_widths=None):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4F81BD")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
    ]))
    return table
def styled_table(data, col_widths=None):
    table = Table(data, colWidths=col_widths, repeatRows=1)

    HEADER_BG = colors.HexColor("#589edf")   # xanh đậm
    ROW_BG = colors.HexColor("#e3ebf4")      # xanh nhạt
    GRID_COLOR = colors.HexColor("#9bb7d4")

    table.setStyle(TableStyle([
        # Header
        ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Times"),
        ("FONTSIZE", (0,0), (-1,0), 11),

        # Body
        ("FONTNAME", (0,1), (-1,-1), "Times"),
        ("FONTSIZE", (0,1), (-1,-1), 10.5),
        ("BACKGROUND", (0,1), (-1,-1), ROW_BG),

        # Layout
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.6, GRID_COLOR),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
    ]))

    return table



def load_company_table_top8(page_width):
    # ===== 1. Load TOP 8 allocation =====
    alloc_df = pd.read_csv(
        DATA_POWERBI_DIR / "asset_allocation_current.csv"
    )

    top_alloc = (
        alloc_df
        .sort_values("allocation_weight", ascending=False)
        .head(8)
        .copy()
    )

    top_alloc["allocation_pct"] = top_alloc["allocation_weight"] * 100

    # ===== 2. Load company info =====
    company_df = pd.read_excel(
        r"C:\Users\ASUS\fintech-project\data_raw\company.xlsx"
    )

    merged = company_df.merge(
        top_alloc[["symbol", "allocation_pct"]],
        left_on="Mã chứng khoán",
        right_on="symbol",
        how="inner"
    ).sort_values("allocation_pct", ascending=False)

    # ===== 3. PARAGRAPH STYLE (QUAN TRỌNG NHẤT) =====
    name_style = ParagraphStyle(
        name="CompanyName",
        fontName="Times",
        fontSize=10.2,
        leading=13.5,              # khoảng cách dòng (RẤT QUAN TRỌNG)
        alignment=TA_LEFT,
        spaceBefore=2,
        spaceAfter=2,
        leftIndent=0,
        rightIndent=0,
        wordWrap="CJK"              #  BẮT BUỘC để wrap tiếng Việt
    )

    header_style = ParagraphStyle(
        name="Header",
        fontName="Times",
        fontSize=11,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    header_right = ParagraphStyle(
        name="HeaderRight",
        fontName="Times",
        fontSize=11,
        textColor=colors.white,
        alignment=TA_RIGHT
    )

    # ===== 4. BUILD TABLE DATA =====
    data = [
        [
            Paragraph("Tên doanh nghiệp", header_style),
            Paragraph("Mã chứng khoán", header_style),
            Paragraph("Tỷ trọng (%)", header_right),
        ]
    ]

    for _, r in merged.iterrows():
        data.append([
            Paragraph(r["Tên đầy đủ"], name_style),   #  AUTO XUỐNG DÒNG
            r["Mã chứng khoán"],
            f"{r['allocation_pct']:.2f}%",
        ])

    # ===== 5. TABLE =====
    table = Table(
        data,
        colWidths=[
            page_width * 0.28,   # Tên doanh nghiệp (giảm chút)
            page_width * 0.09,   # Mã CK
            page_width * 0.11    # Tỷ trọng
        ],
        repeatRows=1
    )


    # ===== 6. STYLE CHUẨN PORTFOLIO VISUALIZER =====
    style = [
    # ================= HEADER =================
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#3b6edc")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("VALIGN", (0,0), (-1,0), "MIDDLE"),

    # Font header (KHÔNG bold – dùng size + màu)
    ("FONTNAME", (0,0), (-1,0), "Times"),
    ("FONTSIZE", (0,0), (-1,0), 11),

    # Padding HEADER (thoáng, không sát lề)
    ("LEFTPADDING", (0,0), (0,0), 16),     # cột Tên DN
    ("LEFTPADDING", (1,0), (-1,0), 14),
    ("RIGHTPADDING", (0,0), (-1,0), 16),
    ("TOPPADDING", (0,0), (-1,0), 10),
    ("BOTTOMPADDING", (0,0), (-1,0), 10),

    # Header alignment
    ("ALIGN", (0,0), (0,0), "LEFT"),
    ("ALIGN", (1,0), (1,0), "CENTER"),
    ("ALIGN", (2,0), (2,0), "RIGHT"),

    # ================= BODY =================
    ("FONTNAME", (0,1), (-1,-1), "Times"),
    ("FONTSIZE", (0,1), (-1,-1), 10.2),

    # Alignment body
    ("ALIGN", (0,1), (0,-1), "LEFT"),
    ("ALIGN", (1,1), (1,-1), "CENTER"),
    ("ALIGN", (2,1), (2,-1), "RIGHT"),

    # Padding BODY
    ("LEFTPADDING", (0,1), (0,-1), 16),    # cột Tên DN (quan trọng)
    ("LEFTPADDING", (1,1), (-1,-1), 14),
    ("RIGHTPADDING", (2,1), (2,-1), 16),
    ("TOPPADDING", (0,1), (-1,-1), 8),
    ("BOTTOMPADDING", (0,1), (-1,-1), 8),

    # ================= GRID =================
    ("GRID", (0,0), (-1,-1), 0.6, colors.HexColor("#9bb7d4")),
    ("VALIGN", (0,1), (-1,-1), "MIDDLE"),
]

    # Zebra rows
    for i in range(1, len(data)):
        bg = colors.whitesmoke if i % 2 == 1 else colors.HexColor("#eaf2ff")
        style.append(("BACKGROUND", (0,i), (-1,i), bg))

    table.setStyle(TableStyle(style))
    return table


# ============================
# PREP DATA FOR GEMINI
# ============================
# =====================================
# PREPARE DATA FOR GEMINI COMMENTARY
# =====================================
def prepare_gemini_inputs(
    perf_metrics: dict,
    risk_metrics: dict,
    frontier_metrics: dict,
    corr_summary: dict,
    annual_returns_df: pd.DataFrame
):
    return {
        "performance": {
            "CAGR": perf_metrics.get("cagr"),
            "Sharpe Ratio": perf_metrics.get("sharpe_ratio"),
            "Max Drawdown": perf_metrics.get("max_drawdown"),
            "Final Value": perf_metrics.get("end_balance"),
        },

        "risk": {
            "Volatility": perf_metrics.get("volatility_annual"),
            "VaR 5%": perf_metrics.get("var_5"),
            "CVaR 5%": perf_metrics.get("cvar_5"),
            "Max Drawdown": perf_metrics.get("max_drawdown"),
        },

        "annual_returns": (
            annual_returns_df
            .replace([np.inf, -np.inf], None)
            .where(pd.notnull(annual_returns_df), None)
            .to_dict(orient="records")
        ),


        "efficient_frontier": frontier_metrics,

        "correlation": {
            k: float(v) if v is not None else None
            for k, v in corr_summary.items()
        },

    }


def load_performance_metrics():
    df = pd.read_csv(DATA_REPORT_DIR / "performance_extended.csv")
    df.columns = df.columns.str.lower()

    def parse(v):
        if isinstance(v, str):
            v = v.replace("%", "").strip()
        try:
            return float(v)
        except:
            return None

    df["value"] = df["value"].apply(parse)
    return dict(zip(df["metric"], df["value"]))


def load_annual_returns_df():
    return pd.read_csv(DATA_POWERBI_DIR / "annual_returns.csv")


def load_frontier_metrics():
    df = pd.read_csv(DATA_POWERBI_DIR / "asset_risk_return.csv")
    ml = df.iloc[0]
    return {
        "Expected Return": ml["expected_return"],
        "Volatility": ml["volatility"],
        "Sharpe Ratio": ml.get("sharpe", None)
    }

def normalize_metric_key(k: str) -> str:
    return (
        k.strip()
        .lower()
        .replace("%", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def load_perf_metrics_from_csv():
    df = pd.read_csv(DATA_REPORT_DIR / "performance_extended.csv")

    # Chuẩn hóa tên cột
    df.columns = df.columns.str.strip().str.lower()

    if not {"metric", "value"}.issubset(df.columns):
        raise ValueError(
            f"performance_extended.csv thiếu cột cần thiết. "
            f"Columns hiện có: {list(df.columns)}"
        )

    return {
        normalize_metric_key(k): v
        for k, v in zip(df["metric"], df["value"])
    }

perf_metrics = load_perf_metrics_from_csv()

# print("DEBUG perf_metrics:", perf_metrics)

# ======================
# BUILD RISK METRICS
# ======================
risk_metrics = {
    "volatility_annual": perf_metrics.get("volatility_annual"),
    "var_5": perf_metrics.get("var_5"),
    "cvar_5": perf_metrics.get("cvar_5"),
    "max_drawdown": perf_metrics.get("max_drawdown"),
}

annual_returns_df = load_annual_returns_df()
frontier_metrics = load_frontier_metrics()
def load_correlation_summary():
    df = pd.read_csv(DATA_REPORT_DIR / "correlation_summary.csv")
    return dict(zip(df["metric"], df["value"]))

corr_summary = load_correlation_summary()


ml_df = pd.read_csv(DATA_POWERBI_DIR / "asset_risk_return.csv")
ml = ml_df.iloc[0]

gemini_inputs = prepare_gemini_inputs(
    perf_metrics=perf_metrics,
    risk_metrics=risk_metrics,
    frontier_metrics=frontier_metrics,
    corr_summary=corr_summary,
    annual_returns_df=annual_returns_df
)

#  GHI ĐÈ ĐÚNG NGUỒN
ml_df = pd.read_csv(DATA_POWERBI_DIR / "asset_risk_return.csv")
ml = ml_df.iloc[0]

gemini_inputs["efficient_frontier"] = {
    "Expected Return": float(ml["expected_return"]),
    "Volatility": float(ml["volatility"]),
    "Sharpe Ratio": float(
        ml["expected_return"] / ml["volatility"]
        if "sharpe_ratio" not in ml or pd.isna(ml.get("sharpe_ratio"))
        else ml["sharpe_ratio"]
    )
}
gemini_inputs["correlation"] = {
    "Average Correlation": corr_summary.get("mean_correlation"),
    "Minimum Correlation": corr_summary.get("min_correlation"),
    "Maximum Correlation": corr_summary.get("max_correlation"),
    "Low Correlation Ratio": corr_summary.get("low_corr_ratio"),
}

COMMENTARY_CACHE = DATA_REPORT_DIR / "commentary.json"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1"}
)

if COMMENTARY_CACHE.exists():
    commentary = json.load(open(COMMENTARY_CACHE, encoding="utf-8"))
else:
    from src.reporting.gemini_commentary import generate_commentary_once

    commentary = generate_commentary_once(
        perf_dict=gemini_inputs["performance"],
        risk_dict=gemini_inputs["risk"],
        frontier_dict=gemini_inputs["efficient_frontier"],
        corr_summary=gemini_inputs["correlation"],
        annual_returns_df=annual_returns_df,
        client=client
    )


    with open(COMMENTARY_CACHE, "w", encoding="utf-8") as f:
        json.dump(commentary, f, ensure_ascii=False, indent=2)
import hashlib

def hash_inputs(d):
    return hashlib.md5(
        json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

COMMENTARY_CACHE = DATA_REPORT_DIR / "commentary.json"
HASH_FILE = DATA_REPORT_DIR / "commentary.hash"

current_hash = hash_inputs(gemini_inputs)

def run_gemini():
    return generate_commentary_once(
        perf_dict=gemini_inputs["performance"],
        risk_dict=gemini_inputs["risk"],
        frontier_dict=gemini_inputs["efficient_frontier"],
        corr_summary=gemini_inputs["correlation"],
        annual_returns_df=annual_returns_df,
        client=client
    )

if COMMENTARY_CACHE.exists() and HASH_FILE.exists():
    if HASH_FILE.read_text(encoding="utf-8") == current_hash:
        commentary = json.load(open(COMMENTARY_CACHE, encoding="utf-8"))
    else:
        commentary = run_gemini()
        json.dump(commentary, open(COMMENTARY_CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        HASH_FILE.write_text(current_hash, encoding="utf-8")
else:
    commentary = run_gemini()
    json.dump(commentary, open(COMMENTARY_CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    HASH_FILE.write_text(current_hash, encoding="utf-8")

# =========================
# VALIDATE GEMINI INPUTS
# =========================
def validate_inputs(d):
    missing = [k for k, v in d.items() if v is None]
    if missing:
        raise ValueError(f" Thiếu dữ liệu Gemini: {missing}")

validate_inputs(gemini_inputs["performance"])
validate_inputs(gemini_inputs["risk"])
validate_inputs(gemini_inputs["efficient_frontier"])


def image_text_row(img_path, text, page_width, styles, img_height=260):
    img = Image(str(img_path), width=(page_width - 80) / 2, height=img_height)
    img.hAlign = "CENTER"

    return Table(
        [[img, Paragraph(text, styles["Normal"])]],
        colWidths=[(page_width - 80) / 2, (page_width - 80) / 2],
        style=[("VALIGN", (0, 0), (-1, -1), "TOP")]
    )
def _build_risk_table_block(page_width, styles, risk_metrics, risk_commentary):

    vol = risk_metrics.get("Volatility") \
          or risk_metrics.get("annual_volatility") \
          or risk_metrics.get("portfolio_volatility")

    mdd = risk_metrics.get("Max Drawdown") or risk_metrics.get("max_drawdown")
    sharpe = risk_metrics.get("Sharpe") or risk_metrics.get("sharpe_ratio")

    if vol is None or mdd is None:
        raise ValueError(" risk_metrics thiếu key quan trọng")

    table_data = [
        ["Chỉ tiêu", "Giá trị"],
        ["Volatility", f"{vol:.4f}"],
        ["Max Drawdown", f"{mdd:.4f}"],
        ["Sharpe Ratio", f"{sharpe:.4f}" if sharpe is not None else "N/A"],
    ]

    table = Table(table_data, colWidths=[page_width*0.4, page_width*0.3])
    ...
    return table, risk_commentary





def build_metrics_from_extended(df):
    df = df.copy()
    df["metric"] = df["metric"].str.lower()

    perf = {}
    risk = {}

    mapping_perf = {
        "cagr": "CAGR",
        "sharpe ratio": "Sharpe Ratio",
        "maximum drawdown": "Max Drawdown",
        "end balance": "Final Value"
    }

    mapping_risk = {
        "volatility (annualized)": "Volatility",
        "downside deviation (monthly)": "Downside Risk",
        "maximum drawdown": "Max Drawdown",
        "historical var (5%)": "VaR 5%",
        "conditional var (5%)": "CVaR 5%"
    }

    for k, v in mapping_perf.items():
        val = df.loc[df["metric"] == k, "value"]
        if not val.empty:
            perf[v] = float(val.iloc[0])

    for k, v in mapping_risk.items():
        val = df.loc[df["metric"] == k, "value"]
        if not val.empty:
            risk[v] = float(val.iloc[0])

    return perf, risk


def build_pdf():

    # ===== GENERATE CHARTS =====
    plot_asset_allocation()
    plot_portfolio_growth()
    plot_drawdown()
    plot_annual_returns()
    plot_risk_return()
    plot_efficient_frontier_reference(
        symbols=symbols,
        portfolio_weights=weights,
        lookback_days=252
    )

    plot_risk_return_metrics()
    plot_return_distribution()

    styles = getSampleStyleSheet()
    styles["Title"].fontName = "Times"
    styles["Title"].fontSize = 18
    styles["Heading2"].fontName = "Times"
    styles["Heading2"].fontSize = 13
    styles["Normal"].fontName = "Times"
    styles["Normal"].fontSize = 11

    PAGE_W, PAGE_H = landscape(A4)



    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=landscape(A4),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    elements = []
    CONTENT_W = PAGE_W - doc.leftMargin - doc.rightMargin

    # ===== TITLE (KHÔNG PAGEBREAK) =====
    elements.append(Paragraph("<b>BÁO CÁO TỐI ƯU DANH MỤC ĐẦU TƯ</b>", styles["Title"]))
    elements.append(Spacer(1, 10))
    elements.append(
        Paragraph(
            f"Dữ liệu cập nhật đến ngày {datetime.now():%d/%m/%Y}",
            styles["Normal"]
        )
    )
    elements.append(Spacer(1, 18))
    elements.append(
    Paragraph(
        f"<i>Các chỉ số được tính từ ngày bắt đầu đầu tư: "
        f"{pd.to_datetime(INVESTMENT_START_DATE).strftime('%d/%m/%Y')}</i>",
        styles["Normal"]
    )
    )
    elements.append(Spacer(1, 6))

    # ===== 1. ASSET ALLOCATION (TRANG 1) =====
    elements.append(Paragraph("Cơ cấu phân bổ tài sản", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    asset_img = Image(
        IMG_DIR / "asset_allocation.png",
        width=PAGE_W * 0.42,
        height=PAGE_H * 0.42
    )

    company_table =  load_company_table_top8(PAGE_W)

    layout_table = Table(
        [
            [
                asset_img,
                company_table
            ]
        ],
        colWidths=[PAGE_W * 0.45, PAGE_W * 0.45]
    )

    layout_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))

    elements.append(layout_table)
    elements.append(PageBreak())
    # ===== 2. TỔNG QUAN DANH MỤC (PV STYLE – CĂN ĐỀU, 1 TRANG) =====
    elements.append(
        Paragraph("Tổng quan danh mục", styles["Heading2"])
    )
    elements.append(Spacer(1, 8))

    # ===== LOAD DATA =====
    df = pd.read_csv(DATA_REPORT_DIR / "performance_extended.csv")
    df.columns = df.columns.str.lower()

    table_data = [["Chỉ tiêu", "Giá trị"]]

    for _, r in df.iterrows():
        key = normalize_metric_key(r["metric"])

        label_vi = METRIC_LABEL_VI.get(
            key,
            r["metric"]   # fallback nếu chưa map
        )

        val = r["value"]
        if pd.isna(val):
            display_val = ""
        elif isinstance(val, (int, float)):
            display_val = f"{val:.4f}"
        else:
            display_val = str(val)

        table_data.append([label_vi, display_val])


    # ===== PALETTE PORTFOLIO VISUALIZER =====
    PV_BLUE_DARK  = colors.HexColor("#3b6edc")
    PV_BLUE_LIGHT = colors.HexColor("#eaf2ff")
    PV_BLUE_GRID  = colors.HexColor("#9bb7d4")
    PV_TEXT_DARK  = colors.HexColor("#1f2933")

    # ===== TABLE (KHÔNG CĂNG FULL, CHỪA LỀ ĐỀU) =====
    risk_table = Table(
        table_data,
        colWidths=[PAGE_W * 0.42, PAGE_W * 0.22],  
        repeatRows=1,
        hAlign="CENTER"
    )

    risk_table.setStyle(TableStyle([
        # Header
        ("BACKGROUND", (0,0), (-1,0), PV_BLUE_DARK),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Times"),
        ("FONTSIZE", (0,0), (-1,0), 11),

        # Body
        ("FONTNAME", (0,1), (-1,-1), "Times"),
        ("FONTSIZE", (0,1), (-1,-1), 10.5),
        ("TEXTCOLOR", (0,1), (-1,-1), PV_TEXT_DARK),

        ("ALIGN", (1,1), (1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

        # Grid
        ("GRID", (0,0), (-1,-1), 0.6, PV_BLUE_GRID),

        # Padding
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))

    # Zebra rows
    for i in range(1, len(table_data)):
        bg = colors.whitesmoke if i % 2 == 1 else PV_BLUE_LIGHT
        risk_table.setStyle(
            TableStyle([("BACKGROUND", (0,i), (-1,i), bg)])
        )

    # ===== ADD TABLE =====
    elements.append(risk_table)
    elements.append(Spacer(1, 14))

    # ===== NHẬN XÉT (NGAY DƯỚI BẢNG – CÙNG TRANG) =====
    elements.append(
        Paragraph(
            commentary["portfolio_overview"],
            styles["Normal"]
        )
    )

    elements.append(PageBreak())


    # =========================
    # 3. DIỄN BIẾN GIÁ TRỊ DANH MỤC (1 TRANG)
    # =========================
    elements.append(
        Paragraph("Diễn biến giá trị danh mục", styles["Heading2"])
    )
    elements.append(Spacer(1, 6))

    # ---- Biểu đồ (fit trang) ----
    portfolio_img = Image(
        IMG_DIR / "portfolio_growth.png",
        width=PAGE_W * 0.88,
        height=PAGE_H * 0.45
    )
    portfolio_img.hAlign = "CENTER"
    elements.append(portfolio_img)

    # ---- Đẩy nhận xét xuống cuối trang ----
    elements.append(Spacer(1, PAGE_H * 0.18))

    elements.append(
        Paragraph(
            commentary["performance"],
            styles["Normal"]
        )
    )

    elements.append(PageBreak())


   # =========================
    # 4. LỢI NHUẬN THEO NĂM (PORTFOLIO VISUALIZER STYLE)
    # =========================
    elements.append(
        Paragraph("Lợi nhuận theo năm", styles["Heading2"])
    )
    elements.append(Spacer(1, 8))

    # ===== LOAD DATA =====
    annual_data = load_table(FILES["annual"])

    # ===== STYLE CHUẨN PV =====
    PV_BLUE_DARK  = colors.HexColor("#3b6edc")
    PV_BLUE_LIGHT = colors.HexColor("#eaf2ff")
    PV_BLUE_GRID  = colors.HexColor("#9bb7d4")
    PV_TEXT_DARK  = colors.HexColor("#1f2933")

    annual_table = Table(
        annual_data,
        repeatRows=1,
        colWidths=[CONTENT_W / len(annual_data[0])] * len(annual_data[0])
    )

    annual_table.setStyle(TableStyle([
        # ===== HEADER =====
        ("BACKGROUND", (0,0), (-1,0), PV_BLUE_DARK),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Times"),
        ("FONTSIZE", (0,0), (-1,0), 11),
        ("ALIGN", (1,0), (-1,0), "CENTER"),

        # ===== BODY =====
        ("FONTNAME", (0,1), (-1,-1), "Times"),
        ("FONTSIZE", (0,1), (-1,-1), 10.2),
        ("TEXTCOLOR", (0,1), (-1,-1), PV_TEXT_DARK),

        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("ALIGN", (0,1), (0,-1), "CENTER"),

        # ===== GRID =====
        ("GRID", (0,0), (-1,-1), 0.6, PV_BLUE_GRID),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

        # ===== PADDING =====
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))

    # ===== ZEBRA ROWS =====
    for i in range(1, len(annual_data)):
        bg = colors.whitesmoke if i % 2 == 1 else PV_BLUE_LIGHT
        annual_table.setStyle(
            TableStyle([("BACKGROUND", (0,i), (-1,i), bg)])
        )

    # ===== WRAP & CENTER TABLE =====
    table_wrapper = Table(
        [[annual_table]],
        colWidths=[CONTENT_W],
        hAlign="CENTER"
    )
    table_wrapper.setStyle(TableStyle([
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))

    elements.append(table_wrapper)

    # ===== ĐẨY NHẬN XÉT XUỐNG CUỐI TRANG =====
    elements.append(Spacer(1, PAGE_H * 0.18))

    elements.append(
        Paragraph(
            commentary["annual_returns"],
            styles["Normal"]
        )
    )

    elements.append(PageBreak())



     # ===== 5. SỤT GIẢM DANH MỤC (FINAL – PORTFOLIO VISUALIZER STYLE) =====
    # ======================
    # PALETTE CHUNG (ĐỒNG BỘ TOÀN REPORT)
    # ======================
    PV_BLUE_DARK  = colors.HexColor("#3b6edc")
    PV_BLUE_MAIN  = colors.HexColor("#6baed6")
    PV_BLUE_LIGHT = colors.HexColor("#eaf2ff")
    PV_BLUE_GRID  = colors.HexColor("#9bb7d4")
    PV_TEXT_DARK  = colors.HexColor("#1f2933")

    elements.append(
        Paragraph(
            "Các giai đoạn sụt giảm lớn của danh mục",
            styles["Heading2"]
        )
    )

    # =========================
    # 1. BIỂU ĐỒ DRAWDOWN (TO – ĐẨY CAO)
    # =========================
    elements.append(Spacer(1, 4))

    drawdown_img = Image(
        IMG_DIR / "drawdown.png",
        width=PAGE_W * 0.90,
        height=PAGE_H * 0.32
    )
    drawdown_img.hAlign = "CENTER"
    elements.append(drawdown_img)

    # =========================
    # 2. COMMENTARY (GIỮA BIỂU ĐỒ & BẢNG)
    # =========================
    elements.append(Spacer(1, 6))

    risk_style = ParagraphStyle(
        "RiskText",
        parent=styles["Normal"],
        textColor=PV_TEXT_DARK,
        leading=14
    )

    elements.append(
        Table(
            [[Paragraph(commentary["risk"], risk_style)]],
            colWidths=[PAGE_W * 0.90],
            rowHeights=[PAGE_H * 0.08]
        )
    )

    elements.append(Spacer(1, 4))

    # =========================
    # 3. BẢNG DRAWDOWN – CHIA 2, DÍNH SÁT, ÉP XUỐNG DƯỚI
    # =========================
    drawdown_data = load_table(FILES["drawdown"])
    header = drawdown_data[0]
    rows = drawdown_data[1:]

    mid = (len(rows) + 1) // 2
    left_rows = rows[:mid]
    right_rows = rows[mid:]

    def styled_drawdown_table(data):
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            # Header
            ("BACKGROUND", (0,0), (-1,0), PV_BLUE_DARK),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Times"),
            ("FONTSIZE", (0,0), (-1,0), 10.5),

            # Body
            ("FONTNAME", (0,1), (-1,-1), "Times"),
            ("FONTSIZE", (0,1), (-1,-1), 10),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

            # Padding bảng con
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),

            #  GRID CHỈ VẼ VIỀN NGOÀI
            ("BOX", (0,0), (-1,-1), 0.6, PV_BLUE_GRID),
            ("INNERGRID", (0,0), (-1,-1), 0.6, PV_BLUE_GRID),
        ]))

        # Zebra rows
        for i in range(1, len(data)):
            bg = colors.whitesmoke if i % 2 == 1 else PV_BLUE_LIGHT
            table.setStyle(
                TableStyle([("BACKGROUND", (0,i), (-1,i), bg)])
            )

        return table



    left_table = styled_drawdown_table([header] + left_rows)
    right_table = styled_drawdown_table([header] + right_rows)

    bottom_table = Table(
        [[left_table, right_table]],
        colWidths=[
            PAGE_W * 0.44,
            PAGE_W * 0.44
        ],
        hAlign="CENTER"   #  CANH GIỮA
    )

    bottom_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),

        #  ZERO PADDING → KHÔNG CÓ KHE
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))


    bottom_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),

        #  KHÔNG CÓ KHOẢNG CÁCH GIỮA 2 BẢNG
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))

    # Đẩy cụm bảng xuống sát đáy trang
    elements.append(Spacer(1, PAGE_H * 0.03))

    elements.append(
        KeepTogether([
            bottom_table
        ])
    )


    elements.append(PageBreak())


    def two_charts_page(img1, txt1, img2, txt2, title):
        elements.append(Paragraph(title, styles["Heading2"]))
        elements.append(Spacer(1, 10))

        img_w = PAGE_W * 0.42
        img_h = PAGE_H * 0.30

        table = Table(
            [
                # HÀNG 1: HÌNH TRÁI – TEXT PHẢI
                [
                    Image(IMG_DIR / img1, width=img_w, height=img_h),
                    Paragraph(txt1, styles["Normal"])
                ],

                # HÀNG 2: TEXT TRÁI – HÌNH PHẢI
                [
                    Paragraph(txt2, styles["Normal"]),
                    Image(IMG_DIR / img2, width=img_w, height=img_h)
                ]
            ],
            colWidths=[PAGE_W * 0.45, PAGE_W * 0.45],
            rowHeights=[PAGE_H * 0.38, PAGE_H * 0.38]
        )

        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        elements.append(table)
        elements.append(PageBreak())
    two_charts_page(
        "risk_return.png",
        str(commentary["correlation"]),

        "efficient_frontier.png",
        str(commentary["frontier"]),

        "Rủi ro và tối ưu danh mục"
    )
    two_charts_page(
        "return_distribution.png",
        commentary["risk"],                 # phân phối → rủi ro
        "risk_return_metrics.png",
        commentary["portfolio_overview"],   # chỉ số → tổng quan
        "Phân phối lợi nhuận và chỉ số rủi ro"
    )



    


    # ===== 7. KHUYẾN NGHỊ =====
    elements.append(Paragraph("Khuyến nghị đầu tư", styles["Heading2"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(commentary["recommendation"], styles["Normal"]))

    doc.build(elements)
    print(" PDF REPORT GENERATED:", OUT_PDF)


if __name__ == "__main__":
    build_pdf()
