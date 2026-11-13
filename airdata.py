"""에어코리아 내려받기 파일 읽기 + 분석 + 예측 (streamlit/plotly 없이 pandas·numpy만 사용)"""
import csv
import io
import re

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 항목 / 등급 기준
# ------------------------------------------------------------------
ITEMS = {  # 키: (표시 이름, 단위)
    "PM10": ("PM10", "㎍/㎥"),
    "PM25": ("PM2.5", "㎍/㎥"),
    "O3": ("오존", "ppm"),
    "NO2": ("이산화질소", "ppm"),
    "CO": ("일산화탄소", "ppm"),
    "SO2": ("아황산가스", "ppm"),
}
KEYS = list(ITEMS)

GRADE_LABELS = ["좋음", "보통", "나쁨", "매우나쁨"]
GRADE_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
GRADE_COLOR_MAP = dict(zip(GRADE_LABELS, GRADE_COLORS))
GRADE_EMOJI = {"좋음": "🔵", "보통": "🟢", "나쁨": "🟠", "매우나쁨": "🔴"}
# 환경부 통합대기환경지수: 좋음/보통/나쁨 상한값, 그 이상은 매우나쁨
GRADE_TH = {
    "PM10": [30, 80, 150],
    "PM25": [15, 35, 75],
    "O3": [0.030, 0.090, 0.150],
    "NO2": [0.030, 0.060, 0.200],
    "CO": [2, 9, 15],
    "SO2": [0.020, 0.050, 0.150],
}

# 서울 25개 구 대표 좌표 (구청 부근, 지도 표시용 근사값)
SEOUL_GU = {
    "강남구": (37.5172, 127.0473), "강동구": (37.5301, 127.1238), "강북구": (37.6396, 127.0257),
    "강서구": (37.5509, 126.8495), "관악구": (37.4784, 126.9516), "광진구": (37.5385, 127.0823),
    "구로구": (37.4954, 126.8874), "금천구": (37.4569, 126.8955), "노원구": (37.6542, 127.0568),
    "도봉구": (37.6688, 127.0471), "동대문구": (37.5744, 127.0396), "동작구": (37.5124, 126.9393),
    "마포구": (37.5663, 126.9019), "서대문구": (37.5791, 126.9368), "서초구": (37.4837, 127.0324),
    "성동구": (37.5634, 127.0369), "성북구": (37.5894, 127.0167), "송파구": (37.5145, 127.1059),
    "양천구": (37.5169, 126.8664), "영등포구": (37.5264, 126.8963), "용산구": (37.5324, 126.9900),
    "은평구": (37.6027, 126.9291), "종로구": (37.5735, 126.9790), "중구": (37.5638, 126.9976),
    "중랑구": (37.6063, 127.0926),
}


def item_title(key: str) -> str:
    name, unit = ITEMS[key]
    return f"{name} ({unit})"


def item_key(name: str):
    """파일 머리글의 항목 이름 -> 내부 키."""
    n = str(name).lower().replace(" ", "").replace("_", "")
    if "pm2.5" in n or "pm25" in n or "초미세" in n:
        return "PM25"
    if "pm10" in n or "미세먼지" in n:
        return "PM10"
    if "오존" in n or n.startswith("o3"):
        return "O3"
    if "이산화질소" in n or n.startswith("no2"):
        return "NO2"
    if "일산화탄소" in n or n == "co" or n.startswith("co("):
        return "CO"
    if "아황산" in n or n.startswith("so2"):
        return "SO2"
    return None


def get_grade(value, key):
    if key not in GRADE_TH or pd.isna(value):
        return None
    for i, t in enumerate(GRADE_TH[key]):
        if value <= t:
            return GRADE_LABELS[i]
    return GRADE_LABELS[-1]


def fmt_value(v, key=None):
    if pd.isna(v):
        return "-"
    if key in ("O3", "NO2", "SO2"):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}"


# ------------------------------------------------------------------
# 파일 읽기
# ------------------------------------------------------------------
def to_number(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return out.where(out >= 0)  # '-', '점검중', 음수 결측코드는 모두 결측 처리


def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


_TIME_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})[: ]\s*(\d{1,2})$")


def parse_airkorea(raw: bytes, name: str = ""):
    """에어코리아 '측정소별 시간 자료' 내려받기 형식을 읽는다. 형식이 다르면 None."""
    lines = decode_bytes(raw).splitlines()
    sample = "\n".join(lines[:15])
    delim = max([",", "\t", ";"], key=sample.count)
    rows = [[c.strip() for c in r] for r in csv.reader(lines, delimiter=delim)]

    hdr = next((i for i, r in enumerate(rows[:15]) if r and r[0].replace(" ", "") in ("날짜", "일시", "측정일시")), None)
    if hdr is None:
        return None
    start = next((i for i in range(hdr + 1, min(len(rows), hdr + 8)) if rows[i] and _TIME_RE.match(rows[i][0])), None)
    if start is None:
        return None

    width = max(len(r) for r in rows[hdr:])

    def pad(r):
        return r + [""] * (width - len(r))

    names = pad(rows[hdr])
    for j in range(1, width):  # 병합 셀로 비어 있는 항목명은 왼쪽 값으로 채움
        if names[j] == "":
            names[j] = names[j - 1]
    between = [pad(r) for r in rows[hdr + 1:start] if r]
    sub_row = next((r for r in between if any("등급" in c for c in r)), None)

    value_idx = {}
    for j in range(1, width):
        key = item_key(names[j])
        if key is None or key in value_idx:
            continue
        sub = sub_row[j] if sub_row else ""
        if sub and "1시간" not in sub:  # 등급 열, 24시간 평균 열 등은 건너뜀
            continue
        value_idx[key] = j
    if not value_idx:
        return None

    data = [pad(r) for r in rows[start:] if r and _TIME_RE.match(r[0])]
    parts = pd.DataFrame([_TIME_RE.match(r[0]).groups() for r in data], columns=["year", "month", "day", "hour"]).astype(int)
    base = pd.to_datetime(parts[["year", "month", "day"]])
    times = base + pd.to_timedelta(parts["hour"], unit="h")  # 24시 -> 다음 날 0시

    title = next((c for r in rows[:hdr] for c in r if c), "")
    station = title.split("(")[0].strip() or re.sub(r"\.csv$", "", name, flags=re.I) or "측정소"
    address = title[title.find("(") + 1:title.rfind(")")].strip() if "(" in title and ")" in title else ""

    df = pd.DataFrame({"일시": times})
    df["측정소"] = station
    df["주소"] = address
    df["가상"] = False
    for key, j in value_idx.items():
        df[key] = to_number(pd.Series([r[j] for r in data]))
    return df


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    for k in KEYS:
        if k not in df.columns:
            df[k] = np.nan
    df["표시명"] = [s + (" (예시)" if g else "") for s, g in zip(df["측정소"], df["가상"])]
    cols = ["측정소", "표시명", "주소", "일시", "가상"] + KEYS
    df = df[cols].drop_duplicates(subset=["측정소", "일시"], keep="last")
    return df.sort_values(["측정소", "일시"]).reset_index(drop=True)


def combine(frames) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        empty = pd.DataFrame({
            "측정소": pd.Series([], dtype=object), "주소": pd.Series([], dtype=object),
            "일시": pd.to_datetime([]), "가상": pd.Series([], dtype=bool),
        })
        return _finish(empty)
    return _finish(pd.concat(frames, ignore_index=True))


def merge_demo(real: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    """실제 데이터가 없는 구만 가상 예시 데이터로 채운다."""
    have = set(real["측정소"]) if len(real) else set()
    extra = demo[~demo["측정소"].isin(have)]
    cols = [c for c in real.columns if c != "표시명"]
    return _finish(pd.concat([real[cols], extra[cols]], ignore_index=True))


def hourly_series(data: pd.DataFrame, station=None) -> pd.DataFrame:
    """측정소 1곳(또는 전체 평균)의 1시간 간격 시계열. 빠진 시간은 결측(NaN)으로 채움."""
    sub = data if station is None else data[data["측정소"] == station]
    ts = sub.groupby("일시")[KEYS].mean().sort_index()
    if ts.empty:
        return ts
    return ts.asfreq("h")


# ------------------------------------------------------------------
# 오염 원인 분석 (광화학, PM2.5/PM10 비율)
# ------------------------------------------------------------------
def _argmax_in(s: pd.Series, lo: int, hi: int):
    sub = s.loc[lo:hi].dropna()
    return int(sub.idxmax()) if len(sub) else None


def photochem_summary(ts: pd.DataFrame):
    """NO2·O3 일중 패턴과 상관. 필요한 항목이 부족하면 None."""
    if "O3" not in ts or "NO2" not in ts or ts["O3"].notna().sum() < 24 or ts["NO2"].notna().sum() < 24:
        return None
    d = pd.DataFrame({"NO2": ts["NO2"] * 1000, "O3": ts["O3"] * 1000})  # ppm -> ppb
    d["Ox"] = d["NO2"] + d["O3"]
    if "CO" in ts:
        d["CO"] = ts["CO"]
    d["hour"] = d.index.hour
    prof = d.groupby("hour")[["NO2", "O3", "Ox"]].mean().reindex(range(24))
    day = d[(d["hour"] >= 10) & (d["hour"] <= 17)]
    out = {
        "frame": d,
        "profile": prof,
        "no2_morning": _argmax_in(prof["NO2"], 5, 10),
        "no2_evening": _argmax_in(prof["NO2"], 16, 22),
        "o3_peak": _argmax_in(prof["O3"], 9, 19),
        "corr_all": d["O3"].corr(d["NO2"]),
        "corr_day": day["O3"].corr(day["NO2"]) if len(day) > 10 else np.nan,
        "corr_co_no2": d["CO"].corr(d["NO2"]) if "CO" in d else np.nan,
        "days": (d.index.max() - d.index.min()).days + 1,
    }

    def cv(s):  # 일중 변동폭 / 평균
        m = s.mean()
        return (s.max() - s.min()) / m if m and not np.isnan(m) else np.nan

    out["cv_o3"], out["cv_ox"] = cv(prof["O3"]), cv(prof["Ox"])
    out["no2_mean"], out["o3_mean"] = prof["NO2"].mean(), prof["O3"].mean()
    out["no2_prom_m"] = prof["NO2"].max() / out["no2_mean"] if out["no2_mean"] else np.nan
    return out


def interpret_photochem(s) -> list:
    msgs = []
    if s["days"] < 3:
        msgs.append(f"⚠️ 데이터 기간이 {s['days']}일뿐이라 아래 해석은 경향을 가늠하는 참고용입니다.")
    m, e = s["no2_morning"], s["no2_evening"]
    if m is not None and e is not None:
        if s["no2_prom_m"] >= 1.1:
            msgs.append(f"NO₂는 오전 {m}시, 저녁 {e}시 무렵에 가장 높습니다. 출퇴근 시간대 자동차 배출의 영향일 가능성이 있습니다.")
        else:
            msgs.append(f"NO₂의 하루 변동이 크지 않습니다(오전 {m}시, 저녁 {e}시가 상대적 최고). 교통 외 요인이나 확산 조건의 영향도 함께 볼 필요가 있습니다.")
    if s["o3_peak"] is not None:
        msgs.append(f"O₃는 {s['o3_peak']}시 무렵에 가장 높습니다. 햇빛이 강한 낮에 광화학 반응으로 생성되는 오존의 전형적인 패턴입니다.")
    if not np.isnan(s["cv_o3"]) and not np.isnan(s["cv_ox"]) and s["cv_ox"] < s["cv_o3"]:
        msgs.append("NO₂+O₃ 합(Ox)의 일중 변동이 O₃ 단독보다 작습니다. 교통량이 많은 시간에 O₃가 낮아지는 것은 NO와의 반응(적정)으로 일부 설명될 수 있습니다.")
    c = s["corr_all"]
    if not np.isnan(c):
        if c <= -0.3:
            msgs.append(f"NO₂와 O₃는 음의 상관(r={c:.2f})입니다. NO₂가 높을수록 O₃가 낮은 경향입니다.")
        elif c >= 0.3:
            msgs.append(f"NO₂와 O₃가 양의 상관(r={c:.2f})입니다. 둘 다 낮·정체 조건에서 함께 오르는 경향일 수 있습니다.")
        else:
            msgs.append(f"NO₂와 O₃의 상관은 약합니다(r={c:.2f}).")
    cc = s["corr_co_no2"]
    if not np.isnan(cc) and cc >= 0.6:
        msgs.append(f"CO와 NO₂가 함께 움직입니다(r={cc:.2f}). 연소·교통 배출원을 공유할 가능성을 시사합니다.")
    return msgs


def pm_ratio(ts: pd.DataFrame):
    if "PM10" not in ts or "PM25" not in ts:
        return None
    r = ts["PM25"] / ts["PM10"].where(ts["PM10"] > 0)
    r = r.where(r <= 1.5)
    return r if r.notna().sum() >= 6 else None


def interpret_ratio(mean: float) -> str:
    if mean >= 0.7:
        return "PM2.5 비중이 높습니다. 연소·2차 생성 등 인위적 오염의 영향이 큰 상태일 가능성이 있습니다."
    if mean <= 0.4:
        return "PM2.5 비중이 낮고 큰 입자가 많습니다. 황사, 도로·공사장 먼지 등의 영향일 가능성이 있습니다."
    return "PM2.5 비중이 중간 수준입니다. 어느 한쪽으로 단정하기는 어렵습니다."


# ------------------------------------------------------------------
# 예측 (numpy 릿지 회귀: 과거 값 + 다른 항목 현재값 + 시간대)
# ------------------------------------------------------------------
def _features(ts: pd.DataFrame, target: str, horizon: int, lags: int):
    full = ts.asfreq("h")
    cols = [c for c in KEYS if c in full and full[c].notna().mean() >= 0.7]
    full = full[cols].ffill(limit=3)
    X = pd.DataFrame(index=full.index)
    for L in range(lags):
        X[f"{target}__과거{L}"] = full[target].shift(L)
    for c in cols:
        if c != target:
            X[f"{c}__현재"] = full[c]
    hr = full.index.hour
    X["시간__sin"] = np.sin(2 * np.pi * hr / 24)
    X["시간__cos"] = np.cos(2 * np.pi * hr / 24)
    y = full[target].shift(-horizon)
    return X, y


def _fit(X: np.ndarray, y: np.ndarray, alpha: float):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return mu, sd, w, ym


def _predict(model, X: np.ndarray):
    mu, sd, w, ym = model
    return ((X - mu) / sd) @ w + ym


def forecast(ts: pd.DataFrame, target: str, horizon: int, lags: int = 6, min_rows: int = 48, alpha: float = 5.0):
    """(결과 dict, 안내 문구). 데이터가 부족하면 (None, 이유)."""
    if target not in ts or ts[target].notna().sum() < min_rows:
        return None, f"{ITEMS[target][0]} 데이터가 부족합니다. 최소 {min_rows}시간 이상의 데이터가 필요합니다."
    X, y = _features(ts, target, horizon, lags)
    ok = X.notna().all(axis=1) & y.notna()
    Xv, yv = X[ok], y[ok]
    if len(Xv) < min_rows:
        return None, f"학습에 쓸 수 있는 연속 데이터가 {len(Xv)}시간뿐입니다(최소 {min_rows}시간). 기간이 더 긴 파일을 쓰거나 예측 시점을 줄여보세요."
    x_last = X.iloc[[-1]]
    if x_last.isna().any(axis=1).iloc[0]:
        return None, "가장 최근 시각에 결측값이 있어 예측을 계산할 수 없습니다."

    n_test = max(12, int(len(Xv) * 0.2))
    Xtr, ytr, Xte, yte = Xv.iloc[:-n_test], yv.iloc[:-n_test], Xv.iloc[-n_test:], yv.iloc[-n_test:]
    m_eval = _fit(Xtr.values, ytr.values, alpha)
    pred_te = np.clip(_predict(m_eval, Xte.values), 0, None)
    persist = Xte[f"{target}__과거0"].values
    mae_m = float(np.mean(np.abs(pred_te - yte.values)))
    mae_p = float(np.mean(np.abs(persist - yte.values)))

    m_all = _fit(Xv.values, yv.values, alpha)
    pred = float(max(_predict(m_all, x_last.values)[0], 0))

    imp = {}
    for col, w in zip(Xv.columns, m_all[2]):
        var, kind = col.split("__")
        if var == "시간":
            label = "시간대"
        elif kind.startswith("과거"):
            label = f"{ITEMS[var][0]} 과거값(최근 {lags}시간)"
        else:
            label = f"{ITEMS[var][0]} 현재값"
        imp[label] = imp.get(label, 0.0) + abs(w)
    imp = pd.Series(imp)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)

    test = pd.DataFrame({"실제": yte.values, "모델 예측": pred_te, "지금 값 그대로": persist}, index=Xte.index + pd.Timedelta(hours=horizon))
    return {
        "pred": pred,
        "current": float(x_last[f"{target}__과거0"].iloc[0]),
        "t_now": x_last.index[0],
        "t_pred": x_last.index[0] + pd.Timedelta(hours=horizon),
        "mae": mae_m,
        "mae_persist": mae_p,
        "gain": (1 - mae_m / mae_p) * 100 if mae_p > 0 else np.nan,
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "importance": imp,
        "test": test,
    }, ""


# ------------------------------------------------------------------
# 가상 예시 데이터 (실제 측정값 아님 - 화면 체험용)
# ------------------------------------------------------------------
def _ar1(rng, n, phi, sigma):
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def make_demo(days: int = 14, end=None, seed: int = 7) -> pd.DataFrame:
    end = pd.Timestamp.now().floor("h") if end is None else pd.Timestamp(end)
    idx = pd.date_range(end=end, periods=days * 24, freq="h")
    hr = idx.hour.values
    n = len(idx)
    common = _ar1(np.random.default_rng(seed), n, 0.97, 1.0)  # 서울 전체가 함께 겪는 대기 정체/청정 흐름
    frames = []
    for k, gu in enumerate(SEOUL_GU):
        rng = np.random.default_rng(seed + 1 + k)
        level = rng.uniform(0.8, 1.25)
        pm25 = np.clip(level * (22 + 4 * common) + _ar1(rng, n, 0.8, 1.5), 2, None)
        ratio = np.clip(0.6 + 0.08 * _ar1(rng, n, 0.9, 0.3), 0.3, 0.85)
        pm10 = pm25 / ratio
        traffic = 14 + 9 * np.exp(-((hr - 8) ** 2) / 8) + 10 * np.exp(-((hr - 19) ** 2) / 12.5)
        no2 = np.clip(traffic * rng.uniform(0.7, 1.3) * (1 + 0.12 * common) + rng.normal(0, 1.5, n), 2, None)  # ppb
        o3 = np.clip(12 + 38 * np.exp(-((hr - 15) ** 2) / 24.5) - 0.35 * (no2 - 14) + rng.normal(0, 2, n), 2, None)  # ppb
        co = np.clip(0.25 + 0.012 * (no2 - 14) + rng.normal(0, 0.02, n), 0.1, None)
        so2 = np.clip(0.003 + rng.normal(0, 0.0003, n), 0.0008, None)
        frames.append(pd.DataFrame({
            "일시": idx, "측정소": gu, "주소": "", "가상": True,
            "PM10": pm10.round(0), "PM25": pm25.round(0),
            "O3": (o3 / 1000).round(4), "NO2": (no2 / 1000).round(4),
            "CO": co.round(2), "SO2": so2.round(4),
        }))
    return _finish(pd.concat(frames, ignore_index=True))
