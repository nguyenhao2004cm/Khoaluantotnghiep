/**
 * Chuẩn format tài chính - Intl.NumberFormat
 */

export const fmtPct = (v: number, maxDecimals = 2): string =>
  new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: 0,
    maximumFractionDigits: maxDecimals,
  }).format(v);

export const fmtNum = (v: number, maxDecimals = 2): string =>
  new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: maxDecimals,
  }).format(v);

export const fmtCurrency = (v: number, currency = "VND"): string =>
  new Intl.NumberFormat("vi-VN", {
    style: "currency",
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(v);

export const fmtCompact = (v: number): string =>
  new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 0,
  }).format(v);
