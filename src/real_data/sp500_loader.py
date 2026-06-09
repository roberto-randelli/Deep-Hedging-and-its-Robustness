"""
S&P 500 real-data utilities for out-of-sample evaluation.

Responsibilities:
  1. get_sp500_tickers()         — scrape current constituent list from Wikipedia
  2. build_test_windows()        — rolling 30-day windows from per-ticker CSV files
  3. compute_entropic_risk()     — entropic risk measure on per-path hedging errors

Data layout expected on disk (written by Notebook 01):
    data/spy_prices.csv          — SPY daily closes, index=date, column "Close"
    data/sp500_prices/<TICKER>.csv — per-ticker closes, same format

Rolling window construction:
    For each ticker:
      - Load closing prices for the test period
      - Extract consecutive T-day windows (step = 1 day)
      - Normalise each window so that window[0] = S0_norm
      - Drop windows containing any NaN or zero price
    All valid windows are stacked into a single (M_total, T+1) tensor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def get_sp500_tickers() -> list[str]:
    return [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "ABNB",
    "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN",
    "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI",
    "AON", "APA", "APO", "AAPL", "AMAT", "APP", "APTV", "ACGL", "ADM", "ARES", "ANET",
    "AJG", "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL",
    "BAC", "BAX", "BDX", "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BNY", "BA",
    "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP", "CHRW", "CDNS",
    "CPT", "CPB", "COF", "CAH", "CCL", "CARR", "CVNA", "CASY", "CAT", "CBOE", "CBRE",
    "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD",
    "CIEN", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO", "CTSH",
    "COHR", "COIN", "CL", "CMCSA", "FIX", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT",
    "GLW", "CPAY", "CTVA", "CSGP", "COST", "CRH", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR",
    "DRI", "DDOG", "DVA", "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG",
    "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "ETN", "EBAY",
    "SATS", "ECL", "EIX", "EW", "EA", "ELV", "EME", "EMR", "ETR", "EOG", "EQT", "EFX",
    "EQIX", "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC", "EXE", "EXPE", "EXPD",
    "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FDXF", "FIS", "FITB",
    "FSLR", "FE", "FISV", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT",
    "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL",
    "GDDY", "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HPE", "HLT", "HD",
    "HON", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX",
    "ITW", "INCY", "IR", "PODD", "INTC", "IBKR", "ICE", "IFF", "IP", "INTU", "ISRG", "IVZ",
    "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "KVUE", "KDP",
    "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX",
    "LVS", "LDOS", "LEN", "LII", "LLY", "LIN", "LYV", "LMT", "L", "LOW", "LULU", "LITE",
    "LYB", "MTB", "MPC", "MAR", "MRSH", "MLM", "MAS", "MA", "MKC", "MCD", "MCK", "MDT",
    "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "TAP", "MDLZ",
    "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM", "NWSA",
    "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA",
    "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG",
    "PLTR", "PANW", "PSKY", "PH", "PAYX", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX",
    "PNW", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC",
    "PSA", "PHM", "PWR", "QCOM", "DGX", "Q", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF",
    "RSG", "RMD", "RVTY", "HOOD", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM",
    "SBAC", "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV",
    "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY",
    "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA", "TXN", "TPL",
    "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL",
    "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO",
    "VEEV", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX", "VRT", "VTRS", "VICI", "V", "VST",
    "VMC", "WRB", "GWW", "WAB", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL",
    "WST", "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM", "ZBRA",
    "ZBH", "ZTS"]


def build_test_windows(
    prices_dir: str | Path,
    T: int = 30,
    S0_norm: float = 10.0,
    min_price: float = 0.01,
) -> tuple[torch.Tensor, list[str]]:
    """
    Build rolling T-day price windows from per-ticker CSV files.

    For each CSV file in prices_dir:
      - Read the "Close" column (index = date strings)
      - Extract all consecutive windows of length T+1
      - Normalise each window: S_window = S_window / S_window[0] * S0_norm
      - Drop any window containing NaN, zero, or price < min_price

    Args:
        prices_dir: Path to directory containing <TICKER>.csv files.
        T:          Window length in trading days (default 30).
        S0_norm:    Normalised starting price (default 10.0).
        min_price:  Minimum price threshold; windows with any price below
                    this value are discarded.

    Returns:
        S_real:        Float32 tensor of shape (M_total, T+1).
        ticker_labels: List of length M_total; each entry is the ticker symbol
                       for the corresponding window (for sector breakdown).
    """
    prices_dir = Path(prices_dir)
    csv_files  = sorted(prices_dir.glob("*.csv"))

    windows_list: list[np.ndarray] = []
    labels_list:  list[str]        = []

    for csv_path in csv_files:
        ticker = csv_path.stem
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if "Close" not in df.columns:
                continue
            prices = df["Close"].dropna().values.astype(np.float64)
        except Exception:
            continue

        n = len(prices)
        if n < T + 1:
            continue

        for start in range(n - T):
            window = prices[start : start + T + 1]
            if np.any(np.isnan(window)) or np.any(window < min_price):
                continue
            window_norm = window / window[0] * S0_norm
            windows_list.append(window_norm)
            labels_list.append(ticker)

    if not windows_list:
        return torch.empty(0, T + 1, dtype=torch.float32), []

    S_real = torch.tensor(np.stack(windows_list, axis=0), dtype=torch.float32)
    return S_real, labels_list


def compute_entropic_risk(
    errors: torch.Tensor,
    lamb: float = 1.0,
) -> float:
    """
    Compute the entropic risk measure on a vector of hedging errors.

    The entropic risk measure is:
        ρ_λ(X) = inf_ω { ω + (1/λ) · E[exp(-λ(X + ω))] - 1/λ }

    where X = C_T - PnL is the per-path hedging error (positive = loss).

    The optimal ω satisfies E[exp(-λ(X + ω*))] = 1, giving the closed-form:
        ω* = -log(E[exp(-λ·X)]) / λ
        ρ_λ = ω* + (1/λ)·E[exp(-λ(X + ω*))] - 1/λ
             = ω* + 1/λ - 1/λ
             = ω*
             = -(1/λ) · log(E[exp(-λ·X)])

    Numerically stable via log-sum-exp:
        ρ_λ = -(1/λ) · (log(mean(exp(-λ·X))))
            = -(1/λ) · (LSE(-λ·X) - log(M))

    Args:
        errors: 1-D tensor of hedging errors X = C_T - PnL.
        lamb:   Risk-aversion parameter λ > 0.

    Returns:
        Scalar entropic risk measure (float). Lower is better.
    """
    x = -lamb * errors.float()
    log_mean_exp = torch.logsumexp(x, dim=0) - torch.log(torch.tensor(len(errors), dtype=torch.float32))
    return float(-log_mean_exp / lamb)
