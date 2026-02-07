# -*- coding: utf-8 -*-
"""VERONICA Utils Package"""

from .auth import check_auth, require_auth, logout
from .common import (
    KST, norm_inst_id, parse_symbol, extract_iso_date_to_str,
    extract_iso_date_to_date, yyyymmdd_to_mdy_str, yyyymmdd_to_date,
    normalize_party_list, normalize_quote_list, calculate_month_difference,
    make_pair_symbol, resolve_trade_utc_date, read_csv_safely,
    optimize_dataframe, to_csv_bytes, AppConfig
)
from .okx_api import (
    fetch_okx_tickers_bulk_spot, fetch_okx_ticker_price,
    get_batch_current_prices_okx, get_okx_daily_close_for_date,
    get_batch_okx_closes, debug_fetch_okx_ticker,
    build_current_price_debug_table_okx
)
from .options import (
    to_okx_inst_id, get_close_prices_okx, get_ohlcv_data_okx,
    compute_rolling_volatility, compute_ema, forecast_volatility,
    black_scholes_price
)
