import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import airdata as ad

st.set_page_config(page_title="서울 대기질 분석 대시보드", page_icon="🌫️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEOUL_CENTER = dict(lat=37.5665, lon=126.978)


# ------------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_cached(raw: bytes, name: str):
    return ad.parse_airkorea(raw, name)


@st.cache_data(show_spinner=False)
def demo_cached():
    return ad.make_demo()


def load_all(uploads):
    frames, used, skipped = [], [], []
    paths = sorted(set(glob.glob(os.path.join(BASE_DIR, "*.csv")) + glob.glob(os.path.join(BASE_DIR, "data", "*.csv"))))
    sources = []
    for p in paths:
        with open(p, "rb") as f:
            sources.append((os.path.basename(p), f.read()))
    for u in uploads or []:
        sources.append((u.name, u.getvalue()))
    for name, raw in sources:
        try:
            df = parse_cached(raw, name)
        except Exception:  # noqa: BLE001
            df = None
        if df is None or df.empty:
            skipped.append(name)
        else:
            frames.append(df)
            used.append(f"{name} → {df['측정소'].iloc[0]}")
    return ad.combine(frames), used, skipped


st.title("🌫️ 서울 대기질 분석 대시보드")

st.sidebar.header("⚙️ 데이터")
uploads = st.sidebar.file_uploader("에어코리아 CSV 올리기 (여러 개 가능)", type=["csv"], accept_multiple_files=True)
use_demo = st.sidebar.checkbox("가상 예시 데이터로 빈 구 채우기", value=False, help="실제 파일이 없는 구를 가상의 데이터로 채워 화면을 체험해볼 수 있습니다. 실제 측정값이 아닙니다.")

data, used_files, skipped_files = load_all(uploads)
if use_demo:
    data = ad.merge_demo(data, demo_cached())

if data.empty:
    st.info(
        "읽어온 측정소 데이터가 없습니다.\n\n"
        "- 에어코리아에서 받은 CSV를 깃허브 저장소의 `data` 폴더(또는 `main.py`와 같은 위치)에 올리거나\n"
        "- 왼쪽 사이드바에서 CSV를 직접 올려주세요.\n"
        "- 먼저 화면만 보고 싶다면 사이드바의 '가상 예시 데이터' 체크박스를 켜보세요."
    )
    if skipped_files:
        st.warning("읽지 못한 파일: " + ", ".join(skipped_files))
    st.stop()

real_n = int((~data.groupby("측정소")["가상"].first().astype(bool)).sum())
all_n = data["측정소"].nunique()
st.sidebar.caption(f"측정소 {all_n}곳 (실제 {real_n}곳, 가상 예시 {all_n - real_n}곳)")
with st.sidebar.expander("읽은 파일 목록"):
    for line in used_files:
        st.write("✅ " + line)
    for name in skipped_files:
        st.write("⚠️ 형식을 인식하지 못함: " + name)

if use_demo and all_n > real_n:
    st.warning("⚠️ '(예시)'가 붙은 측정소는 가상으로 만든 데이터입니다. 실제 측정값이 아니니 분석 결과를 사실로 인용하지 마세요.")

display = data.drop_duplicates("측정소").set_index("측정소")["표시명"].to_dict()
stations = sorted(display)
default_station = stations.index("중구") if "중구" in stations else 0
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 분석할 측정소")
station = st.sidebar.selectbox("측정소", stations, index=default_station, format_func=lambda s: display[s])
st.sidebar.caption("원인 분석·예측·상세 탭에 적용됩니다.")

t_min, t_max = data["일시"].min(), data["일시"].max()
st.caption(f"데이터 기간: {t_min:%Y-%m-%d %H시} ~ {t_max:%Y-%m-%d %H시} · 시각은 에어코리아 표기 기준(해당 시 종료 시점)입니다.")

tab_map, tab_cause, tab_fc, tab_detail = st.tabs(["🗺️ 서울 지도 비교", "🔬 오염 원인 분석", "🔮 농도 예측", "📈 측정소 상세"])


# ------------------------------------------------------------------
# 탭 1: 서울 지도 비교
# ------------------------------------------------------------------
def draw_map(snap_map, item):
    fn = getattr(px, "scatter_map", None)
    common = dict(
        lat="lat", lon="lon", color="등급", text="표시명", hover_name="표시명",
        hover_data={"lat": False, "lon": False, "등급": True, "값": True},
        color_discrete_map=ad.GRADE_COLOR_MAP, category_orders={"등급": ad.GRADE_LABELS},
        zoom=9.6, center=SEOUL_CENTER, height=560,
    )
    if fn is not None:
        fig = fn(snap_map, map_style="open-street-map", **common)
    else:  # 오래된 plotly 대비
        fig = px.scatter_mapbox(snap_map, mapbox_style="open-street-map", **common)
    fig.update_traces(marker=dict(size=22, opacity=0.85), textposition="top center")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, title_text="등급"))
    return fig


with tab_map:
    c1, c2 = st.columns([1, 2])
    item = c1.selectbox("항목", ad.KEYS, index=1, format_func=ad.item_title, key="map_item")
    times = pd.DatetimeIndex(sorted(data.loc[data[item].notna(), "일시"].unique()))
    if len(times) == 0:
        st.warning("이 항목의 측정값이 없습니다.")
    else:
        labels = [t.strftime("%Y-%m-%d %H시") for t in times]
        if len(labels) > 1:
            picked = c2.select_slider("기준 시각", options=labels, value=labels[-1])
        else:  # 시각이 하나뿐이면 슬라이더를 만들 수 없음
            picked = labels[0]
            c2.caption(f"기준 시각: {picked}")
        t_sel = times[labels.index(picked)]

        snap = data.loc[data["일시"] == t_sel, ["측정소", "표시명", item]].dropna(subset=[item]).rename(columns={item: "값"})
        snap["등급"] = snap["값"].apply(lambda v: ad.get_grade(v, item))
        if snap.empty:
            st.warning("이 시각에 측정값이 있는 측정소가 없습니다.")
        else:
            coords = pd.DataFrame([(g, la, lo) for g, (la, lo) in ad.SEOUL_GU.items()], columns=["측정소", "lat", "lon"])
            snap_map = snap.merge(coords, on="측정소", how="inner")
            left, right = st.columns([3, 2])
            with left:
                if len(snap_map):
                    st.plotly_chart(draw_map(snap_map, item), width="stretch")
                    st.caption("원 색깔은 환경부 통합대기환경지수 등급입니다. 해당 시각에 값이 없는 측정소는 표시되지 않습니다.")
                else:
                    st.info("서울 25개 구에 해당하는 측정소가 없어 지도는 표시하지 않습니다. 오른쪽 순위 그래프를 확인하세요.")
            with right:
                rank = snap.sort_values("값", ascending=False).reset_index(drop=True)
                fig_r = px.bar(
                    rank.iloc[::-1], x="값", y="표시명", orientation="h", color="등급",
                    color_discrete_map=ad.GRADE_COLOR_MAP, category_orders={"등급": ad.GRADE_LABELS},
                )
                fig_r.update_layout(
                    template="plotly_white", title=f"측정소별 {ad.ITEMS[item][0]} 순위",
                    xaxis_title=ad.item_title(item), yaxis_title="",
                    height=max(360, 26 * len(rank) + 120), margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_r, width="stretch")

            if len(rank) >= 2:
                m1, m2, m3 = st.columns(3)
                m1.metric("가장 높은 곳", rank["표시명"].iloc[0], ad.fmt_value(rank["값"].iloc[0], item), delta_color="off")
                m2.metric("가장 낮은 곳", rank["표시명"].iloc[-1], ad.fmt_value(rank["값"].iloc[-1], item), delta_color="off")
                m3.metric("서울 평균", ad.fmt_value(rank["값"].mean(), item), f"{len(rank)}곳 평균", delta_color="off")

            st.markdown("##### 측정소 시간 추이 비교")
            options = sorted(snap["표시명"].tolist())
            default_pick = rank["표시명"].head(3).tolist()
            pick = st.multiselect("비교할 측정소", options, default=default_pick)
            if pick:
                sub = data[data["표시명"].isin(pick)]
                fig_l = px.line(sub, x="일시", y=item, color="표시명")
                fig_l.update_layout(
                    template="plotly_white", height=380, yaxis_title=ad.item_title(item),
                    xaxis_title="", hovermode="x unified", margin=dict(t=20, b=10),
                    legend=dict(orientation="h", y=1.1, title_text=""),
                )
                st.plotly_chart(fig_l, width="stretch")

    if real_n < 3 and not use_demo:
        st.info(
            "지금은 실제 측정소 파일이 적어 지도가 단출합니다. 에어코리아에서 같은 형식으로 다른 구 파일을 더 받아 `data` 폴더에 올려보세요. "
            "파일 이름은 달라도 되지만 서로 겹치지 않게 바꿔 주세요(예: 강남구.csv). 측정소 이름은 파일 안의 제목 줄에서 읽습니다."
        )

# ------------------------------------------------------------------
# 선택 측정소 시계열 (탭 2~4 공용)
# ------------------------------------------------------------------
ts = ad.hourly_series(data, station)
is_demo_station = bool(data.loc[data["측정소"] == station, "가상"].iloc[0])

# ------------------------------------------------------------------
# 탭 2: 오염 원인 분석
# ------------------------------------------------------------------
with tab_cause:
    st.subheader(f"{display[station]} 오염 원인 분석")
    if is_demo_station:
        st.warning("선택한 측정소는 가상 예시 데이터입니다.")
    scope = st.radio("분석 대상", ["선택한 측정소", "전체 측정소 평균"], horizontal=True) if all_n > 1 else "선택한 측정소"
    ts_c = ts if scope == "선택한 측정소" else ad.hourly_series(data, None)

    summ = ad.photochem_summary(ts_c)
    st.markdown("#### 1. 광화학 반응: 이산화질소(NO₂)와 오존(O₃)")
    if summ is None:
        st.info("NO₂·O₃ 데이터가 24시간 이상 있어야 분석할 수 있습니다.")
    else:
        prof = summ["profile"]
        fig_p = go.Figure()
        for col_name, label, color, dash in [
            ("NO2", "이산화질소 (NO₂)", "#ef4444", "solid"),
            ("O3", "오존 (O₃)", "#3b82f6", "solid"),
            ("Ox", "NO₂ + O₃ 합 (Ox)", "#6b7280", "dot"),
        ]:
            fig_p.add_trace(go.Scatter(
                x=prof.index, y=prof[col_name], mode="lines+markers", name=label,
                line=dict(color=color, width=2, dash=dash), marker=dict(size=5),
            ))
        fig_p.update_layout(
            template="plotly_white", height=380, title="시간대별 평균 농도 (하루 중 변화)",
            xaxis=dict(title="시각(시)", dtick=2), yaxis_title="평균 농도 (ppb)",
            hovermode="x unified", legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10),
        )
        a, b = st.columns(2)
        a.plotly_chart(fig_p, width="stretch")

        sc = summ["frame"].dropna(subset=["NO2", "O3"])
        fig_s = px.scatter(sc, x="NO2", y="O3", color="hour", color_continuous_scale="Viridis", opacity=0.65,
                           labels={"NO2": "NO₂ (ppb)", "O3": "O₃ (ppb)", "hour": "시각(시)"})
        fig_s.update_layout(template="plotly_white", height=380, title="NO₂와 O₃의 관계 (점 하나 = 1시간)", margin=dict(t=50, b=10))
        b.plotly_chart(fig_s, width="stretch")

        for line in ad.interpret_photochem(summ):
            st.markdown("- " + line)

    st.markdown("#### 2. 초미세먼지 비중 (PM2.5 / PM10)")
    ratio = ad.pm_ratio(ts_c)
    if ratio is None:
        st.info("PM10과 PM2.5 데이터가 충분하지 않아 비율을 계산할 수 없습니다.")
    else:
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Scatter(x=ratio.index, y=ratio.values, mode="lines", name="PM2.5/PM10", line=dict(color="#8b5cf6", width=1.8)))
        fig_r2.add_hline(y=0.7, line_dash="dash", line_color="#ef4444", annotation_text="0.7 이상: 인위적 오염 비중 큼")
        fig_r2.add_hline(y=0.4, line_dash="dash", line_color="#f59e0b", annotation_text="0.4 이하: 큰 입자(황사·먼지) 비중 큼")
        fig_r2.update_layout(
            template="plotly_white", height=340, yaxis=dict(title="PM2.5 ÷ PM10", range=[0, 1.05]),
            xaxis_title="", margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_r2, width="stretch")
        rm = float(ratio.mean())
        st.markdown(f"- 기간 평균 비율은 **{rm:.2f}**입니다. {ad.interpret_ratio(rm)}")
    st.caption("※ 위 해석은 농도 패턴으로 추정한 가능성일 뿐, 오염원을 확정하는 결과는 아닙니다.")

# ------------------------------------------------------------------
# 탭 3: 농도 예측
# ------------------------------------------------------------------
with tab_fc:
    st.subheader(f"{display[station]} 농도 예측")
    if is_demo_station:
        st.warning("선택한 측정소는 가상 예시 데이터라 예측 결과에 의미가 없습니다. 화면 체험용으로만 보세요.")
    f1, f2 = st.columns(2)
    target = f1.selectbox("예측할 항목", ad.KEYS, index=1, format_func=ad.item_title, key="fc_item")
    horizon = f2.selectbox("예측 시점", [1, 3, 6, 12, 24], index=1, format_func=lambda h: f"{h}시간 뒤", key="fc_h")

    res, msg = ad.forecast(ts, target, horizon)
    if res is None:
        st.info(msg)
    else:
        grade = ad.get_grade(res["pred"], target)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{res['t_pred']:%m월 %d일 %H시} 예상 {ad.ITEMS[target][0]}", ad.fmt_value(res["pred"], target),
                  f"현재 {ad.fmt_value(res['current'], target)}", delta_color="off")
        k2.metric("예상 등급", f"{ad.GRADE_EMOJI.get(grade, '')} {grade}" if grade else "-")
        k3.metric("최근 검증 평균오차", ad.fmt_value(res["mae"], target), f"검증 {res['n_test']}시간", delta_color="off")
        gain = res["gain"]
        k4.metric("'지금 값 그대로' 대비", f"{gain:+.0f}%" if not np.isnan(gain) else "-", "양수일수록 모델이 더 정확", delta_color="off")

        if not np.isnan(gain) and gain <= 0:
            st.warning("이 조건에서는 모델이 '지금 값 그대로 유지'라는 단순 예측보다 낫지 않습니다. 예측값은 참고만 하세요.")
        elif not np.isnan(gain) and gain < 15:
            st.info("모델이 단순 예측보다 조금만 나은 수준입니다. 데이터가 더 쌓이면 좋아질 수 있습니다.")

        tdf = res["test"]
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=tdf.index, y=tdf["실제"], mode="lines", name="실제값", line=dict(color="#111827", width=2)))
        fig_t.add_trace(go.Scatter(x=tdf.index, y=tdf["모델 예측"], mode="lines", name="모델 예측", line=dict(color="#3b82f6", width=2)))
        fig_t.add_trace(go.Scatter(x=tdf.index, y=tdf["지금 값 그대로"], mode="lines", name="지금 값 그대로", line=dict(color="#9ca3af", width=1.5, dash="dot")))
        fig_t.add_trace(go.Scatter(x=[res["t_pred"]], y=[res["pred"]], mode="markers", name="이번 예측",
                                   marker=dict(color="#ef4444", size=13, symbol="star")))
        fig_t.update_layout(
            template="plotly_white", height=400, title="검증 구간: 모델이 과거를 얼마나 잘 맞혔나",
            yaxis_title=ad.item_title(target), xaxis_title="", hovermode="x unified",
            legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10),
        )
        g1, g2 = st.columns([3, 2])
        g1.plotly_chart(fig_t, width="stretch")

        imp = res["importance"].head(8).iloc[::-1]
        fig_i = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "영향 비중(%)", "y": ""})
        fig_i.update_traces(marker_color="#10b981")
        fig_i.update_layout(template="plotly_white", height=400, title="예측에 영향을 준 변수", margin=dict(t=50, b=10))
        g2.plotly_chart(fig_i, width="stretch")

        st.caption(
            f"방법: 최근 6시간의 {ad.ITEMS[target][0]} 값, 다른 항목의 현재값, 시간대를 입력으로 하는 릿지 회귀입니다. "
            f"전체 {res['n_train'] + res['n_test']}시간 중 앞 {res['n_train']}시간으로 학습하고 뒤 {res['n_test']}시간으로 검증했습니다. "
            "기상(풍속·강수 등) 정보가 없어 급격한 변화는 맞히기 어렵습니다."
        )

# ------------------------------------------------------------------
# 탭 4: 측정소 상세
# ------------------------------------------------------------------
with tab_detail:
    st.subheader(f"{display[station]} 상세")
    sd = data[data["측정소"] == station]
    addr = sd["주소"].iloc[0]
    if addr:
        st.caption(f"📍 {addr}")

    shown = [k for k in ad.KEYS if ts[k].notna().any()]
    for start in range(0, len(shown), 3):
        cols = st.columns(3)
        for col, k in zip(cols, shown[start:start + 3]):
            valid = ts[k].dropna()
            latest, avg = valid.iloc[-1], valid.mean()
            col.metric(f"{ad.ITEMS[k][0]} ({ad.ITEMS[k][1]})", ad.fmt_value(latest, k),
                       f"{latest - avg:+.3g} (평균 대비)", delta_color="inverse")
            g = ad.get_grade(latest, k)
            col.caption(f"{ad.GRADE_EMOJI[g]} {g} · 평균 {ad.fmt_value(avg, k)} · 최고 {ad.fmt_value(valid.max(), k)} · 최저 {ad.fmt_value(valid.min(), k)}")

    if shown:
        n_items = len(shown)
        n_cols = 2 if n_items >= 4 else 1
        n_rows = int(np.ceil(n_items / n_cols))
        fig_d = make_subplots(
            rows=n_rows, cols=n_cols, shared_xaxes=True, vertical_spacing=0.12 if n_rows > 1 else 0.1,
            horizontal_spacing=0.06, subplot_titles=[ad.item_title(k) for k in shown],
        )
        palette = px.colors.qualitative.Set2
        for i, k in enumerate(shown):
            r, c = i // n_cols + 1, i % n_cols + 1
            y = ts[k]
            fig_d.add_trace(go.Scatter(
                x=y.index, y=y.values, mode="lines+markers" if len(y) <= 100 else "lines", marker=dict(size=5),
                line=dict(width=1.8, color=palette[i % len(palette)]), name=ad.ITEMS[k][0], connectgaps=False,
                hovertemplate="%{x|%Y-%m-%d %H시}<br>%{y:.4g}<extra>" + ad.ITEMS[k][0] + "</extra>",
            ), row=r, col=c)
            th = ad.GRADE_TH[k]
            lows = [0] + th
            highs = th + [max(float(y.max()) * 1.05, th[-1] * 1.3)]
            for lo, hi, band_color in zip(lows, highs, ad.GRADE_COLORS):
                fig_d.add_hrect(y0=lo, y1=hi, fillcolor=band_color, opacity=0.10, line_width=0, row=r, col=c)
        fig_d.update_layout(
            height=max(380, 300 * n_rows), template="plotly_white", showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
        )
        st.plotly_chart(fig_d, width="stretch")
        st.caption("배경색: 🔵 좋음  🟢 보통  🟠 나쁨  🔴 매우나쁨 (환경부 통합대기환경지수 기준)")

    with st.expander("📄 원본 데이터 보기 / 내려받기"):
        table = sd[["일시"] + ad.KEYS].rename(columns={k: ad.item_title(k) for k in ad.KEYS}).sort_values("일시", ascending=False)
        st.dataframe(table.head(500), width="stretch")
        st.download_button("CSV 내려받기", data=table.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{station}_대기질.csv", mime="text/csv")
