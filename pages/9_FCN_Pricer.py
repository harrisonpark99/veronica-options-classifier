import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="FCN Pricer", page_icon="🔷", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-box {
        background: #1c1f2e; border-radius: 10px; padding: 16px;
        border: 1px solid #2d3148; margin: 4px;
    }
    .metric-label { color: #8892b0; font-size: 12px; font-weight: 600; letter-spacing: 1px; }
    .metric-value { color: #ccd6f6; font-size: 24px; font-weight: 700; }
    .metric-value.green  { color: #64ffda; }
    .metric-value.red    { color: #ff6b6b; }
    .metric-value.yellow { color: #ffd700; }
    .metric-value.blue   { color: #7eb3ff; }
    .section-title {
        color: #ccd6f6; font-size: 15px; font-weight: 700;
        border-bottom: 2px solid #64ffda; padding-bottom: 6px; margin: 20px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 데이터 클래스
# ══════════════════════════════════════════════════════════════════
@dataclass
class FCNParams:
    S0: float
    K_pct: float        # Strike (e.g. 0.90)
    KI_pct: float       # Knock-In barrier (e.g. 0.65)
    KO_pct: float       # Autocall barrier (e.g. 1.00)
    tenor_m: int        # 만기 (개월)
    nc_months: int      # Non-call period (개월)
    coupon_pa: float    # 연간 쿠폰 (e.g. 0.20)
    sigma: float        # 연환산 변동성
    r: float            # 무위험이자율
    notional: float = 1_000_000.0
    n_paths: int = 50_000

    @property
    def T(self):      return self.tenor_m / 12
    @property
    def K(self):      return self.S0 * self.K_pct
    @property
    def KI(self):     return self.S0 * self.KI_pct
    @property
    def KO(self):     return self.S0 * self.KO_pct


def _replace(p: FCNParams, **kwargs) -> FCNParams:
    """FCNParams 필드 일부만 교체해서 새 인스턴스 반환"""
    d = {f: getattr(p, f) for f in
         ["S0","K_pct","KI_pct","KO_pct","tenor_m","nc_months",
          "coupon_pa","sigma","r","notional","n_paths"]}
    d.update(kwargs)
    return FCNParams(**d)


# ══════════════════════════════════════════════════════════════════
# Black-Scholes 헬퍼
# ══════════════════════════════════════════════════════════════════
def bs_put(S, K, r, sigma, T):
    if T <= 0: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def di_put_european(S, K, H, r, sigma, T):
    if T <= 0: return max(K-S, 0.0) if S <= H else 0.0
    d1_H = (np.log(S/H) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2_H = d1_H - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2_H) - S*norm.cdf(-d1_H)


# ══════════════════════════════════════════════════════════════════
# Monte Carlo 코어 — Common Random Numbers 지원
# ══════════════════════════════════════════════════════════════════
def _generate_Z(n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    """공통 난수 행렬 생성 (Greeks 계산 시 노이즈 제거용)"""
    return np.random.default_rng(seed).standard_normal((n_paths, n_steps))

def _mc_from_Z(Z: np.ndarray, p: FCNParams) -> float:
    """
    사전 생성된 Z로 FCN NPV(비율) 계산.
    Daily KI + Monthly Autocall.
    """
    n, n_steps = Z.shape
    dt = p.T / n_steps

    log_ret = (p.r - 0.5*p.sigma**2)*dt + p.sigma*np.sqrt(dt)*Z
    S_paths = p.S0 * np.exp(np.cumsum(log_ret, axis=1))
    S_T = S_paths[:, -1]

    # Daily KI
    daily_ki = S_paths.min(axis=1) < p.KI

    # Monthly KO
    obs_idx   = [min(i*21 - 1, n_steps-1) for i in range(1, p.tenor_m+1)]
    call_idx  = obs_idx[p.nc_months:]
    call_times = [(p.nc_months + i + 1) / 12 for i in range(len(call_idx))]

    autocalled    = np.zeros(n, dtype=bool)
    autocall_time = np.full(n, np.nan)
    if call_idx:
        ko_hit  = S_paths[:, call_idx] >= p.KO
        any_ko  = ko_hit.any(axis=1)
        fki     = np.where(any_ko, ko_hit.argmax(axis=1), 0)
        autocalled    = any_ko
        autocall_time = np.where(any_ko, [call_times[i] for i in fki], np.nan)

    # Payoff
    ko_payoff = 1.0 + p.coupon_pa * autocall_time
    ko_disc   = np.where(autocalled, np.exp(-p.r * autocall_time), 1.0)

    mat_principal = np.where(daily_ki & (S_T < p.K), S_T / p.K, 1.0)
    mat_payoff    = mat_principal + p.coupon_pa * p.T
    mat_disc      = np.exp(-p.r * p.T)

    npv = np.where(autocalled, ko_payoff*ko_disc, mat_payoff*mat_disc)
    return float(np.mean(npv))


def run_mc(params: FCNParams, seed: int = 42) -> dict:
    """전체 통계 반환 (UI용)"""
    n_steps = params.tenor_m * 21
    Z = _generate_Z(params.n_paths, n_steps, seed)
    dt = params.T / n_steps

    log_ret = (params.r - 0.5*params.sigma**2)*dt + params.sigma*np.sqrt(dt)*Z
    S_paths = params.S0 * np.exp(np.cumsum(log_ret, axis=1))
    S_T = S_paths[:, -1]

    daily_ki = S_paths.min(axis=1) < params.KI

    obs_idx    = [min(i*21-1, n_steps-1) for i in range(1, params.tenor_m+1)]
    call_idx   = obs_idx[params.nc_months:]
    call_times = [(params.nc_months+i+1)/12 for i in range(len(call_idx))]

    autocalled    = np.zeros(params.n_paths, dtype=bool)
    autocall_time = np.full(params.n_paths, np.nan)
    if call_idx:
        ko_hit = S_paths[:, call_idx] >= params.KO
        any_ko = ko_hit.any(axis=1)
        fki    = np.where(any_ko, ko_hit.argmax(axis=1), 0)
        autocalled    = any_ko
        autocall_time = np.where(any_ko, [call_times[i] for i in fki], np.nan)

    ko_payoff = 1.0 + params.coupon_pa * autocall_time
    ko_disc   = np.where(autocalled, np.exp(-params.r * autocall_time), 1.0)
    mat_principal = np.where(daily_ki & (S_T < params.K), S_T / params.K, 1.0)
    mat_payoff    = mat_principal + params.coupon_pa * params.T
    mat_disc      = np.exp(-params.r * params.T)

    npv_paths = np.where(autocalled, ko_payoff*ko_disc, mat_payoff*mat_disc)
    npv_mean  = float(np.mean(npv_paths))
    npv_std   = float(np.std(npv_paths) / np.sqrt(params.n_paths))

    ko_by_month = {}
    for i, t in enumerate(call_times):
        ko_by_month[f"M{params.nc_months+i+1}"] = float(np.mean(autocalled & (autocall_time == t)))

    loss_mask = ~autocalled & daily_ki & (S_T < params.K)
    return {
        "npv":          npv_mean,
        "npv_std":      npv_std,
        "npv_95ci":     (npv_mean - 1.96*npv_std, npv_mean + 1.96*npv_std),
        "ko_prob":      float(np.mean(autocalled)),
        "avg_ko_time":  float(np.nanmean(autocall_time)) if autocalled.any() else np.nan,
        "ki_prob":      float(np.mean(daily_ki & ~autocalled)),
        "loss_prob":    float(np.mean(loss_mask)),
        "avg_loss":     float(np.mean(1.0 - S_T[loss_mask]/params.K)) if loss_mask.any() else 0.0,
        "ko_by_month":  ko_by_month,
        "S_T":          S_T,
        "daily_ki":     daily_ki,
        "autocalled":   autocalled,
        "di_put_bs":    di_put_european(params.S0, params.K, params.KI, params.r, params.sigma, params.T),
        "vanilla_put":  bs_put(params.S0, params.K, params.r, params.sigma, params.T),
    }


def _bisect(Z: np.ndarray, params: FCNParams, field: str,
            lo: float, hi: float, target_npv: float = 1.0) -> float:
    """공통난수 Z로 field를 이진탐색, NPV = target_npv 만족하는 값 반환"""
    for _ in range(60):
        mid = (lo + hi) / 2
        p = _replace(params, **{field: mid})
        if _mc_from_Z(Z, p) < target_npv:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fair_coupon_pa(params: FCNParams, target_npv: float = 1.0) -> float:
    Z = _generate_Z(params.n_paths, params.tenor_m * 21, seed=42)
    return _bisect(Z, params, "coupon_pa", 0.0, 5.0, target_npv)


def fair_strike(params: FCNParams, target_npv: float = 1.0) -> float:
    """목표 쿠폰 고정 → NPV = target 이 되는 Strike (K_pct) 역산
    Strike ↑ → 투자자 위험 ↑ → 쿠폰 ↑ → at-par NPV 충족
    Strike 범위: KI_pct + 0.01 ~ 1.05
    """
    Z = _generate_Z(params.n_paths, params.tenor_m * 21, seed=42)
    lo, hi = params.KI_pct + 0.01, 1.05
    # NPV는 Strike에 대해 단조 감소 (Strike 낮을수록 손실 가능성 낮아 NPV 높음)
    # → lo=KI+0.01에서 NPV 최대, hi=1.05에서 NPV 최소
    if _mc_from_Z(Z, _replace(params, K_pct=lo)) < target_npv:
        return lo  # 이미 lo에서도 NPV < target → 해 없음
    return _bisect(Z, params, "K_pct", lo, hi, target_npv)


def fair_ki(params: FCNParams, target_npv: float = 1.0) -> float:
    """목표 쿠폰 고정 → NPV = target 이 되는 KI Barrier (KI_pct) 역산
    KI ↑ → 위험 ↑ → 쿠폰 증가 효과 → at-par NPV 충족하려면 KI 낮춰야
    KI 범위: 0.30 ~ K_pct - 0.01
    """
    Z = _generate_Z(params.n_paths, params.tenor_m * 21, seed=42)
    lo, hi = 0.30, params.K_pct - 0.01
    # NPV는 KI에 대해 단조 감소 (KI 낮을수록 위험 낮아 NPV 높음)
    if _mc_from_Z(Z, _replace(params, KI_pct=hi)) > target_npv:
        return hi  # 해 없음 (쿠폰이 너무 낮아 최대 KI에서도 NPV > target)
    return _bisect(Z, params, "KI_pct", lo, hi, target_npv)


def fair_ko(params: FCNParams, target_npv: float = 1.0) -> float:
    """목표 쿠폰 고정 → NPV = target 이 되는 KO Barrier (KO_pct) 역산
    KO ↓ → autocall 빈번 → 기대 쿠폰 수입 감소 → NPV 감소
    KO 범위: 0.90 ~ 1.30
    """
    Z = _generate_Z(params.n_paths, params.tenor_m * 21, seed=42)
    return _bisect(Z, params, "KO_pct", 0.90, 1.30, target_npv)


# ══════════════════════════════════════════════════════════════════
# Greeks — Central Difference (Common Random Numbers)
# ══════════════════════════════════════════════════════════════════
def _bump_spot(p: FCNParams, S0_new: float) -> FCNParams:
    """
    S0 변경 시 배리어를 달러 고정으로 유지.
    FCN은 발행 시 달러 배리어가 확정되므로 spot 변동 시 pct만 역산.
    """
    K_fix, KI_fix, KO_fix = p.K, p.KI, p.KO
    return _replace(p, S0=S0_new,
                    K_pct=K_fix/S0_new, KI_pct=KI_fix/S0_new, KO_pct=KO_fix/S0_new)


def compute_greeks(params: FCNParams, seed: int = 42) -> dict:
    """
    Delta, Vega, Theta, Rho — 중앙차분 + 공통난수

    Delta : dNPV/dS0  (% NPV per 1% S move, 달러 배리어 고정)
    Vega  : dNPV/dσ   (% NPV per 1% vol move)
    Theta : dNPV/dT   (% NPV per 1 day)
    Rho   : dNPV/dr   (% NPV per 10bps rate move)
    """
    n_steps = params.tenor_m * 21
    Z = _generate_Z(params.n_paths, n_steps, seed)

    eps_S = params.S0 * 0.01          # 1% of spot
    eps_v = 0.01                       # 1% vol
    eps_T = 1 / 365                    # 1 day
    eps_r = 0.001                      # 10bps

    # Delta — 달러 배리어 고정, spot만 변화
    npv_su = _mc_from_Z(Z, _bump_spot(params, params.S0 + eps_S))
    npv_sd = _mc_from_Z(Z, _bump_spot(params, params.S0 - eps_S))
    delta_pct = (npv_su - npv_sd) / 2   # NPV 변화(비율) per 1% S move

    # Vega
    npv_vu = _mc_from_Z(Z, _replace(params, sigma=params.sigma + eps_v))
    npv_vd = _mc_from_Z(Z, _replace(params, sigma=params.sigma - eps_v))
    vega_pct = (npv_vu - npv_vd) / 2    # NPV 변화(비율) per 1% vol move

    # Theta (tenor 단축 방향)
    if params.T > eps_T + 1/365:
        # tenor_m 조정 대신 S0에 T 변화 반영: 직접 _mc_from_Z에 T 변화 필요
        # → T를 줄인 새 Z 생성 (n_steps 달라지므로 별도 계산)
        p_td = _replace(params, tenor_m=max(params.tenor_m, 1))
        n_steps_td = max(params.tenor_m*21 - 1, 1)
        Z_td = _generate_Z(params.n_paths, n_steps_td, seed)
        dt_td = (params.T - eps_T) / n_steps_td

        log_ret_td = (p_td.r - 0.5*p_td.sigma**2)*dt_td + p_td.sigma*np.sqrt(dt_td)*Z_td
        S_paths_td = p_td.S0 * np.exp(np.cumsum(log_ret_td, axis=1))
        S_T_td     = S_paths_td[:, -1]
        ki_td      = S_paths_td.min(axis=1) < p_td.KI

        obs_idx_td  = [min(i*21-1, n_steps_td-1) for i in range(1, p_td.tenor_m+1)]
        call_idx_td = obs_idx_td[p_td.nc_months:]
        ct_td       = [(p_td.nc_months+i+1)/12 for i in range(len(call_idx_td))]
        aut_td      = np.zeros(params.n_paths, dtype=bool)
        aut_time_td = np.full(params.n_paths, np.nan)
        if call_idx_td:
            koh = S_paths_td[:, call_idx_td] >= p_td.KO
            ak  = koh.any(axis=1)
            fk  = np.where(ak, koh.argmax(axis=1), 0)
            aut_td = ak
            aut_time_td = np.where(ak, [ct_td[i] for i in fk], np.nan)
        T_td = params.T - eps_T
        ko_p  = 1.0 + p_td.coupon_pa * aut_time_td
        ko_d  = np.where(aut_td, np.exp(-p_td.r*aut_time_td), 1.0)
        mp    = np.where(ki_td & (S_T_td < p_td.K), S_T_td/p_td.K, 1.0)
        mp2   = mp + p_td.coupon_pa * T_td
        md    = np.exp(-p_td.r * T_td)
        npv_td = float(np.mean(np.where(aut_td, ko_p*ko_d, mp2*md)))

        npv_base = _mc_from_Z(Z, params)
        theta_pct = (npv_td - npv_base)   # NPV 변화(비율) per 1 day (time decay)
    else:
        theta_pct = np.nan

    # Rho
    npv_ru = _mc_from_Z(Z, _replace(params, r=params.r + eps_r))
    npv_rd = _mc_from_Z(Z, _replace(params, r=params.r - eps_r))
    rho_pct = (npv_ru - npv_rd) / 2     # NPV 변화(비율) per 10bps

    return {
        "delta": delta_pct * 100,   # % NPV per 1% S
        "vega":  vega_pct  * 100,   # % NPV per 1% vol
        "theta": theta_pct * 100 if not np.isnan(theta_pct) else np.nan,  # % NPV per day
        "rho":   rho_pct   * 100,   # % NPV per 10bps
    }


# ══════════════════════════════════════════════════════════════════
# 시나리오 Heatmap — Spot shock × Vol
# ══════════════════════════════════════════════════════════════════
def compute_heatmap(params: FCNParams,
                    spot_shocks: np.ndarray,
                    vol_levels: np.ndarray,
                    seed: int = 99) -> np.ndarray:
    """
    NPV(%) 2D grid: rows=vol_levels, cols=spot_shocks
    달러 배리어 고정: spot이 변해도 K/KI/KO 달러 수준은 발행 시 그대로 유지.
    """
    n_vol, n_spot = len(vol_levels), len(spot_shocks)
    grid = np.zeros((n_vol, n_spot))
    n_steps = params.tenor_m * 21
    Z = _generate_Z(params.n_paths, n_steps, seed)

    for i, v in enumerate(vol_levels):
        for j, shock in enumerate(spot_shocks):
            S0_new = params.S0 * (1 + shock)
            p = _bump_spot(_replace(params, sigma=v), S0_new)
            grid[i, j] = _mc_from_Z(Z, p) * 100
    return grid


# ══════════════════════════════════════════════════════════════════
# 시장 데이터
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def fetch_spot(ticker: str) -> Optional[float]:
    try:
        df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        return float(df["Close"].dropna().iloc[-1]) if df is not None and len(df) > 0 else None
    except:
        return None

@st.cache_data(ttl=300)
def fetch_hv(ticker: str, window: int = 30) -> Optional[float]:
    try:
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        if df is not None and len(df) > window:
            lr = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            return float(lr.iloc[-window:].std() * np.sqrt(365))
    except:
        return None


# ══════════════════════════════════════════════════════════════════
# UI 헬퍼
# ══════════════════════════════════════════════════════════════════
def mbox(label, val, color=""):
    cls = f"metric-value {color}" if color else "metric-value"
    return (f'<div class="metric-box"><div class="metric-label">{label}</div>'
            f'<div class="{cls}">{val}</div></div>')


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
st.title("🔷 FCN Pricer")
st.caption("Fixed Coupon Note · Single Underlying · GBM Monte Carlo · Daily KI + Autocall + Greeks")

# ── 사이드바 ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 기초자산")
    asset = st.selectbox("기초자산", ["BTC-USD", "ETH-USD", "직접 입력"])

    spot_auto, hv_auto = None, None
    if asset != "직접 입력":
        with st.spinner("시장 데이터 조회 중..."):
            spot_auto = fetch_spot(asset)
            hv_auto   = fetch_hv(asset, window=30)

    S0 = st.number_input("현재 가격 S₀",
                         value=float(round(spot_auto or 80_000.0, 2)),
                         min_value=1.0, step=100.0, format="%.2f")

    st.markdown("---")
    st.markdown("### 🎯 Solve Mode")
    solve_mode = st.radio(
        "구할 파라미터",
        ["쿠폰 계산 (Strike → Coupon)",
         "Strike 역산 (Coupon → Strike)",
         "KI 역산 (Coupon → KI Barrier)",
         "KO 역산 (Coupon → KO Level)"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### FCN 조건")

    # 각 모드에서 고정 입력 / 역산 출력 구분
    solve_coupon = solve_mode.startswith("쿠폰")
    solve_strike = solve_mode.startswith("Strike")
    solve_ki     = solve_mode.startswith("KI")
    solve_ko     = solve_mode.startswith("KO")

    # Strike: 쿠폰 계산 모드에서만 슬라이더 (나머지는 입력)
    if solve_strike:
        st.info("🎯 Strike를 역산합니다", icon=None)
        K_pct = 0.90  # placeholder (계산 결과로 대체)
    else:
        K_pct = st.slider("Strike (%)", min_value=70, max_value=105, value=90, step=1) / 100

    # KI
    if solve_ki:
        st.info("🎯 KI Barrier를 역산합니다", icon=None)
        KI_pct = 0.65  # placeholder
    else:
        KI_pct = st.slider("KI Barrier (%)", min_value=40, max_value=85, value=65, step=1) / 100

    # KO
    if solve_ko:
        st.info("🎯 KO Level을 역산합니다", icon=None)
        KO_pct = 1.00  # placeholder
    else:
        KO_pct = st.slider("KO (Autocall) (%)", min_value=90, max_value=120, value=100, step=1) / 100

    tenor_m   = st.slider("만기 (개월)", min_value=1, max_value=24, value=6, step=1)
    nc_months = st.slider("Non-call period (개월)", min_value=0, max_value=tenor_m-1, value=0, step=1)

    # 쿠폰: 쿠폰 계산 모드일 때만 역산, 나머지는 목표 쿠폰 입력
    if solve_coupon:
        st.info("🎯 Fair Coupon을 역산합니다", icon=None)
        coupon_pa_pct = 20  # placeholder
    else:
        coupon_pa_pct = st.slider("목표 쿠폰 (% p.a.)", min_value=0, max_value=100, value=20, step=1)

    st.markdown("---")
    st.markdown("### 시장 파라미터")
    sigma = st.slider("변동성 σ (% p.a.)", min_value=10, max_value=200,
                      value=int((hv_auto or 0.65)*100), step=1) / 100
    r     = st.slider("무위험이자율 r (%)", min_value=0, max_value=15, value=5, step=1) / 100

    st.markdown("---")
    st.markdown("### 시뮬레이션")
    n_paths  = st.select_slider("경로 수", options=[10_000, 30_000, 50_000, 100_000], value=50_000)
    notional = st.number_input("명목원금 (USD)", value=1_000_000, step=100_000)

    st.markdown("---")
    st.markdown("### 시나리오 설정")
    run_greeks   = st.checkbox("Greeks 계산", value=True)
    run_heatmap  = st.checkbox("Spot×Vol Heatmap", value=True)
    heatmap_paths = st.select_slider("Heatmap 경로 수", options=[5_000, 10_000, 20_000], value=10_000)

    run_btn = st.button("▶ 계산 실행", use_container_width=True, type="primary")

# ── 파라미터 요약 ────────────────────────────────────────────────
col_l, col_r = st.columns([3, 2])
T = tenor_m / 12

with col_l:
    st.markdown('<div class="section-title">파라미터 요약</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "항목": ["기초자산", "S₀", "Strike K", "KI Barrier", "KO (Autocall)",
                 "만기", "Non-call", "쿠폰 p.a.", "σ", "r"],
        "값": [
            asset, f"${S0:,.2f}",
            f"${S0*K_pct:,.2f}  ({K_pct*100:.0f}%)",
            f"${S0*KI_pct:,.2f}  ({KI_pct*100:.0f}%)",
            f"${S0*KO_pct:,.2f}  ({KO_pct*100:.0f}%)",
            f"{tenor_m}개월", f"{nc_months}개월",
            f"{coupon_pa_pct:.0f}% p.a.", f"{sigma*100:.0f}%", f"{r*100:.1f}%",
        ]
    }), use_container_width=True, hide_index=True)

with col_r:
    st.markdown('<div class="section-title">해석적 참고 (BS)</div>', unsafe_allow_html=True)
    vanilla = bs_put(S0, S0*K_pct, r, sigma, T)
    di_bs   = di_put_european(S0, S0*K_pct, S0*KI_pct, r, sigma, T)
    st.markdown(mbox("Vanilla Put",       f"${vanilla:,.0f}"),          unsafe_allow_html=True)
    st.markdown(mbox("European DI Put",   f"${di_bs:,.0f}", "yellow"),  unsafe_allow_html=True)
    st.markdown(mbox("DI / Vanilla",      f"{di_bs/vanilla*100:.1f}%" if vanilla > 0 else "N/A"),
                unsafe_allow_html=True)
    st.caption("※ 해석적 가격은 European KI 기준 (Daily KI 가격보다 낮음)")


# ── 계산 실행 ────────────────────────────────────────────────────
if run_btn:
    # ── Solve Mode: 역산 먼저 수행 ─────────────────────────────
    # 역산 파라미터는 placeholder로 넣었으므로 실제 값으로 교체
    base_params = FCNParams(
        S0=S0, K_pct=K_pct, KI_pct=KI_pct, KO_pct=KO_pct,
        tenor_m=tenor_m, nc_months=nc_months,
        coupon_pa=coupon_pa_pct/100, sigma=sigma, r=r,
        notional=notional, n_paths=n_paths,
    )

    solved_label = ""
    solved_value = ""

    if solve_coupon:
        with st.spinner("Fair Coupon 역산 중..."):
            fc_pa = fair_coupon_pa(base_params)
        params = _replace(base_params, coupon_pa=fc_pa)
        solved_label = "🎯 Fair Coupon (p.a.)"
        solved_value = f"{fc_pa*100:.2f}%"

    elif solve_strike:
        with st.spinner("Fair Strike 역산 중..."):
            fk = fair_strike(base_params)
        params = _replace(base_params, K_pct=fk)
        solved_label = "🎯 Fair Strike"
        solved_value = f"${S0*fk:,.0f}  ({fk*100:.1f}%)"
        K_pct = fk   # 이후 차트용

    elif solve_ki:
        with st.spinner("Fair KI Barrier 역산 중..."):
            fki = fair_ki(base_params)
        params = _replace(base_params, KI_pct=fki)
        solved_label = "🎯 Fair KI Barrier"
        solved_value = f"${S0*fki:,.0f}  ({fki*100:.1f}%)"
        KI_pct = fki

    elif solve_ko:
        with st.spinner("Fair KO Level 역산 중..."):
            fko = fair_ko(base_params)
        params = _replace(base_params, KO_pct=fko)
        solved_label = "🎯 Fair KO Level"
        solved_value = f"${S0*fko:,.0f}  ({fko*100:.1f}%)"
        KO_pct = fko

    # 역산 결과 강조 배너
    if solved_label:
        st.success(f"**{solved_label}** = **{solved_value}**  *(at-par NPV 기준)*")

    with st.spinner(f"Monte Carlo {n_paths:,}경로 시뮬레이션 중..."):
        result = run_mc(params)

    # 쿠폰 계산 모드가 아닐 때는 역산된 params로 fair_coupon도 재계산
    if not solve_coupon:
        fc_pa = fair_coupon_pa(params)

    # ── 메인 메트릭 ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Monte Carlo 결과</div>', unsafe_allow_html=True)
    npv_pct = result["npv"] * 100

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(mbox("FCN NPV", f"{npv_pct:.2f}%", "green" if npv_pct>=99.5 else "red"), unsafe_allow_html=True)
    with c2:
        label2 = "Fair Coupon (p.a.)" if not solve_coupon else "🎯 Fair Coupon (p.a.)"
        color2 = "green" if solve_coupon else "yellow"
        st.markdown(mbox(label2, f"{fc_pa*100:.2f}%", color2), unsafe_allow_html=True)
    with c3: st.markdown(mbox("Autocall 확률", f"{result['ko_prob']*100:.1f}%", "blue"), unsafe_allow_html=True)
    with c4:
        avg_ko = result["avg_ko_time"]
        st.markdown(mbox("평균 Autocall 시점", f"{avg_ko*12:.1f}개월" if not np.isnan(avg_ko) else "-", "blue"), unsafe_allow_html=True)

    c5,c6,c7,c8 = st.columns(4)
    ci_lo, ci_hi = result["npv_95ci"]
    with c5:
        label5 = "🎯 Strike K" if solve_strike else "Strike K"
        color5 = "green" if solve_strike else ""
        st.markdown(mbox(label5, f"${params.K:,.0f} ({params.K_pct*100:.1f}%)", color5), unsafe_allow_html=True)
    with c6:
        label6 = "🎯 KI Barrier" if solve_ki else "KI Barrier"
        color6 = "green" if solve_ki else ""
        st.markdown(mbox(label6, f"${params.KI:,.0f} ({params.KI_pct*100:.1f}%)", color6), unsafe_allow_html=True)
    with c7:
        label7 = "🎯 KO Level" if solve_ko else "KO Level"
        color7 = "green" if solve_ko else ""
        st.markdown(mbox(label7, f"${params.KO:,.0f} ({params.KO_pct*100:.1f}%)", color7), unsafe_allow_html=True)
    with c8: st.markdown(mbox("NPV 95% CI", f"[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]"), unsafe_allow_html=True)

    c9,c10,c11 = st.columns(3)
    with c9:  st.markdown(mbox("KI 발동 확률 (생존)",  f"{result['ki_prob']*100:.1f}%",  "red"),  unsafe_allow_html=True)
    with c10: st.markdown(mbox("원금손실 확률",         f"{result['loss_prob']*100:.1f}%", "red"),  unsafe_allow_html=True)
    with c11: st.markdown(mbox("평균 원금손실 (KI 시)", f"{result['avg_loss']*100:.1f}%",  "red"),  unsafe_allow_html=True)

    # ── Greeks ──────────────────────────────────────────────────
    if run_greeks:
        st.markdown('<div class="section-title">Greeks (중앙차분 · 공통난수)</div>', unsafe_allow_html=True)

        with st.spinner("Greeks 계산 중..."):
            greeks = compute_greeks(params)

        g1,g2,g3,g4 = st.columns(4)
        delta_dollar = greeks["delta"] / 100 * notional / 100  # $ per 1% S move
        vega_dollar  = greeks["vega"]  / 100 * notional / 100  # $ per 1% vol move
        theta_dollar = greeks["theta"] / 100 * notional if not np.isnan(greeks["theta"]) else np.nan

        with g1:
            color = "green" if greeks["delta"] > 0 else "red"
            st.markdown(mbox("Delta", f"{greeks['delta']:+.3f}%", color), unsafe_allow_html=True)
            st.caption(f"NPV % per 1% S move | ${delta_dollar:+,.0f} / 1% S")
        with g2:
            color = "red" if greeks["vega"] < 0 else "green"
            st.markdown(mbox("Vega", f"{greeks['vega']:+.3f}%", color), unsafe_allow_html=True)
            st.caption(f"NPV % per 1% σ move | ${vega_dollar:+,.0f} / 1% σ")
        with g3:
            if not np.isnan(greeks["theta"]):
                color = "green" if greeks["theta"] > 0 else "red"
                st.markdown(mbox("Theta", f"{greeks['theta']:+.4f}%", color), unsafe_allow_html=True)
                st.caption(f"NPV % per day | ${theta_dollar:+,.0f} / day")
            else:
                st.markdown(mbox("Theta", "N/A"), unsafe_allow_html=True)
        with g4:
            color = "green" if greeks["rho"] > 0 else "red"
            st.markdown(mbox("Rho", f"{greeks['rho']:+.3f}%", color), unsafe_allow_html=True)
            st.caption("NPV % per 10bps rate move")

        # Greeks 해석 노트
        with st.expander("Greeks 해석"):
            st.markdown(f"""
| Greek | 값 | 의미 |
|---|---|---|
| **Delta** `{greeks['delta']:+.3f}%` | S₀ 1% 상승 시 NPV {greeks['delta']:+.3f}% 변화 | FCN은 Short Put이므로 Delta > 0 (S 상승 = 유리) |
| **Vega** `{greeks['vega']:+.3f}%` | σ 1% 상승 시 NPV {greeks['vega']:+.3f}% 변화 | Short Put이므로 Vega < 0 (변동성 상승 = 불리) |
| **Theta** `{greeks['theta']:+.4f}%` | 하루 경과 시 NPV {greeks['theta']:+.4f}% 변화 | Short Put이므로 Theta > 0 (시간 흐름 = 유리) |
| **Rho** `{greeks['rho']:+.3f}%` | r 10bps 상승 시 NPV {greeks['rho']:+.3f}% 변화 | 금리 상승 → 할인율 증가 효과 |
            """)

    # ── Spot × Vol Heatmap ────────────────────────────────────
    if run_heatmap:
        st.markdown('<div class="section-title">Spot × Vol Heatmap (FCN NPV %)</div>', unsafe_allow_html=True)

        spot_shocks = np.array([-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40])
        vol_levels  = np.array([0.30, 0.50, 0.70, 0.90, 1.10, 1.30, 1.50])

        hm_params = _replace(params, n_paths=heatmap_paths)

        with st.spinner("Heatmap 계산 중..."):
            hm_grid = compute_heatmap(hm_params, spot_shocks, vol_levels)

        x_labels = [f"{int(s*100):+d}%" for s in spot_shocks]
        y_labels = [f"{int(v*100)}%" for v in vol_levels]

        # 색상 중앙: 100% = 원금 보존
        fig_hm = go.Figure(go.Heatmap(
            z=hm_grid,
            x=x_labels,
            y=y_labels,
            colorscale=[
                [0.0,  "#7b0000"],   # 70% 이하: 짙은 빨강
                [0.25, "#ff6b6b"],   # 85%: 빨강
                [0.45, "#ffd700"],   # 95%: 노랑
                [0.55, "#64ffda"],   # 105%: 민트
                [0.75, "#0d7c5f"],   # 115%: 짙은 민트
                [1.0,  "#003d2e"],   # 130%+: 매우 짙음
            ],
            zmid=100,
            text=np.round(hm_grid, 1),
            texttemplate="%{text}%",
            textfont=dict(size=11, color="white"),
            showscale=True,
            colorbar=dict(title="NPV (%)", ticksuffix="%"),
        ))
        fig_hm.add_shape(
            type="line", x0=-0.5, x1=len(x_labels)-0.5,
            y0=list(y_labels).index(f"{int(sigma*100)}%") - 0.5 if f"{int(sigma*100)}%" in y_labels else 0,
            y1=list(y_labels).index(f"{int(sigma*100)}%") + 0.5 if f"{int(sigma*100)}%" in y_labels else 0,
            line=dict(color="white", width=2, dash="dot"),
        )
        fig_hm.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            xaxis_title="Spot Shock", yaxis_title="변동성 σ",
            height=420, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("현재 σ 행이 기준. 셀 값 = FCN NPV (% of notional). 100% = 원금 보존.")

    # ── Autocall 시점 분포 ────────────────────────────────────
    if result["ko_by_month"]:
        st.markdown('<div class="section-title">Autocall 시점 분포</div>', unsafe_allow_html=True)
        ko_df = pd.DataFrame({
            "관찰월": list(result["ko_by_month"].keys()),
            "확률 (%)": [v*100 for v in result["ko_by_month"].values()],
        })
        fig_ko = go.Figure(go.Bar(
            x=ko_df["관찰월"], y=ko_df["확률 (%)"],
            marker_color="rgba(126,179,255,0.7)",
            marker_line_color="rgba(126,179,255,1)", marker_line_width=1,
        ))
        fig_ko.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            xaxis_title="관찰월", yaxis_title="Autocall 확률 (%)",
            height=260, margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_ko, use_container_width=True)

    # ── 만기 가격 분포 ────────────────────────────────────────
    st.markdown('<div class="section-title">만기 가격 분포 (Autocall 미발동 경로)</div>', unsafe_allow_html=True)
    S_T = result["S_T"]
    daily_ki  = result["daily_ki"]
    autocalled = result["autocalled"]
    survived  = ~autocalled

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=S_T[survived & ~daily_ki], nbinsx=80, name="KI 미발동",
                               marker_color="rgba(100,255,218,0.45)", marker_line_width=0))
    fig.add_trace(go.Histogram(x=S_T[survived & daily_ki],  nbinsx=80, name="KI 발동",
                               marker_color="rgba(255,107,107,0.55)", marker_line_width=0))
    for x, label, color, pos in [
        (params.K,  f"Strike {K_pct*100:.0f}%",    "#ffd700", "top right"),
        (params.KI, f"KI {KI_pct*100:.0f}%",        "#ff6b6b", "top left"),
        (params.KO, f"KO {KO_pct*100:.0f}%",        "#7eb3ff", "top right"),
        (S0,        "S₀",                            "#8892b0", "top left"),
    ]:
        fig.add_vline(x=x, line_color=color, line_dash="dash",
                      annotation_text=label, annotation_position=pos,
                      annotation_font_color=color)
    fig.update_layout(
        barmode="overlay", template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis_title="만기 가격 S_T", yaxis_title="경로 수",
        height=340, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Vol Sensitivity ───────────────────────────────────────
    st.markdown('<div class="section-title">변동성 Sensitivity</div>', unsafe_allow_html=True)
    vol_range = np.arange(0.20, 2.01, 0.10)
    npv_list, fc_list, ki_list, ko_list = [], [], [], []
    for v in vol_range:
        p = _replace(params, sigma=v, n_paths=20_000)
        res = run_mc(p, seed=1)
        npv_list.append(res["npv"]*100)
        ki_list.append(res["ki_prob"]*100)
        ko_list.append(res["ko_prob"]*100)
        fc_list.append(fair_coupon_pa(p)*100)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=vol_range*100, y=npv_list, name="FCN NPV (%)",
                              line=dict(color="#64ffda", width=2)))
    fig2.add_trace(go.Scatter(x=vol_range*100, y=fc_list, name="Fair Coupon (% p.a.)",
                              line=dict(color="#ffd700", width=2), yaxis="y2"))
    fig2.add_trace(go.Scatter(x=vol_range*100, y=ki_list, name="KI 확률 (%)",
                              line=dict(color="#ff6b6b", width=2, dash="dot"), yaxis="y2"))
    fig2.add_trace(go.Scatter(x=vol_range*100, y=ko_list, name="Autocall 확률 (%)",
                              line=dict(color="#7eb3ff", width=2, dash="dot"), yaxis="y2"))
    fig2.add_vline(x=sigma*100, line_color="#8892b0", line_dash="dash",
                   annotation_text="현재 σ", annotation_font_color="#8892b0")
    fig2.add_hline(y=100, line_color="#2d3148", line_dash="dot")
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="변동성 σ (%)", yaxis_title="FCN NPV (%)",
        yaxis2=dict(title="Fair Coupon / 확률 (%)", overlaying="y", side="right",
                    titlefont=dict(color="#ffd700"), tickfont=dict(color="#ffd700")),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=360, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("← 사이드바에서 파라미터 입력 후 **▶ 계산 실행** 버튼을 누르세요.")
    st.markdown("""
    **구현 내용:**
    - Daily KI + Monthly Autocall (Non-call period 지원)
    - Fair coupon 역산 (공통난수 이진탐색)
    - **Greeks**: Delta · Vega · Theta · Rho (중앙차분 + 공통난수)
    - **Spot × Vol Heatmap**: 2D NPV 시나리오
    - Autocall 시점 분포 / 만기 가격 분포 / Vol sensitivity
    """)
