# =====================================================
# LOG TRANSFORM VERIFICATION
# vol_20, vol_60 đã được log1p trong build_risk_features.py
# Script này chỉ xác nhận và có thể chạy descriptive để kiểm tra skew/kurtosis
# =====================================================

"""
Log transform đã có trong: src/features/build_risk_features.py

    df["vol_20"] = np.log1p(df["vol_20"])
    df["vol_60"] = np.log1p(df["vol_60"])

Sau khi build risk features, chạy descriptive_risk_features.py để kiểm tra:
- Skew(vol_20) < 10
- Kurtosis(vol_20) < 300
"""
