# app.py
import os
import streamlit as st
import pandas as pd
import pymysql
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
from decimal import Decimal, ROUND_DOWN
# =========================================================
# 0) 페이지 기본 설정
# =========================================================
st.set_page_config(
    page_title="Threat Intel FDS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 0-1) 스타일(CSS) - 카드/그림자/라운드/보라 히어로
# =========================================================
CSS = """
<style>
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Noto Sans KR", "Apple SD Gothic Neo", Arial;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }

.sidebar-title {
  font-weight: 800;
  font-size: 1.05rem;
  margin-bottom: .2rem;
}

.topbar-wrap {
  display:flex;
  align-items:center;
  gap:.55rem;
  margin-bottom:.4rem;
}
.badge {
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  padding:.25rem .55rem;
  border-radius:999px;
  background: rgba(99,102,241,0.12);
  color: #4f46e5;
  font-weight:700;
  font-size: .85rem;
}
.title {
  font-size: 1.85rem;
  font-weight: 900;
  letter-spacing: -0.02em;
  margin: 0;
}
.subtitle {
  color: rgba(31,41,55,0.72);
  margin-top: .15rem;
  margin-bottom: 1.0rem;
}

.kpi-grid { display:flex; gap: 14px; }
.kpi-card {
  flex:1;
  background: #ffffff;
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 6px 18px rgba(15,23,42,0.06);
}
.kpi-label {
  font-size: .82rem;
  color: rgba(31,41,55,0.64);
  margin-bottom: 6px;
  font-weight: 650;
}
.kpi-value {
  font-size: 1.55rem;
  font-weight: 900;
  color: rgba(17,24,39,0.92);
  line-height: 1.1;
}

.panel {
  background: #ffffff;
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 6px 18px rgba(15,23,42,0.06);
}
.panel-title {
  font-size: 1.05rem;
  font-weight: 850;
  margin: 0 0 .3rem 0;
}
.panel-sub {
  color: rgba(31,41,55,0.62);
  font-size: .88rem;
  margin: 0 0 .6rem 0;
}

.hero {
  background: linear-gradient(135deg, rgba(79,70,229,0.98), rgba(99,102,241,0.90));
  border-radius: 22px;
  padding: 18px 18px 16px 18px;
  color: #fff;
  box-shadow: 0 10px 30px rgba(79,70,229,0.25);
  border: 1px solid rgba(255,255,255,0.20);
}
.hero-top {
  display:flex;
  align-items:center;
  gap: .5rem;
  margin-bottom: .2rem;
}
.hero-pill {
  display:inline-flex;
  align-items:center;
  gap:.4rem;
  padding:.2rem .55rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.16);
  font-weight: 800;
  font-size: .8rem;
}
.hero-title {
  font-size: 2.0rem;
  font-weight: 950;
  margin: .25rem 0 .25rem 0;
  letter-spacing: -0.02em;
}
.hero-desc {
  opacity: .92;
  font-size: .92rem;
  line-height: 1.35;
  margin-bottom: .8rem;
}
.hero-metrics {
  display:flex;
  gap: 12px;
  flex-wrap: wrap;
}
.mini {
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 16px;
  padding: 12px 12px;
  min-width: 140px;
}
.mini-label {
  font-size: .72rem;
  opacity: .85;
  font-weight: 800;
  letter-spacing: .02em;
}
.mini-value {
  font-size: 1.2rem;
  font-weight: 950;
  margin-top: 2px;
  word-break: break-all;
}

.news-item {
  padding: 10px 10px;
  border-radius: 14px;
  border: 1px solid rgba(15,23,42,0.06);
  margin-bottom: 10px;
  background: rgba(248,250,252,0.75);
}
.news-tag {
  display:inline-block;
  font-size: .76rem;
  color: #2563eb;
  background: rgba(37,99,235,0.10);
  padding: .12rem .45rem;
  border-radius: 999px;
  font-weight: 750;
}
.news-time {
  color: rgba(31,41,55,0.58);
  font-size: .82rem;
}

.okline {
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  padding:.3rem .55rem;
  border-radius: 999px;
  background: rgba(16,185,129,0.12);
  color: rgba(6,95,70,0.95);
  font-weight: 800;
  font-size: .85rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# 1) MySQL 연결 정보
# =========================================================
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "zxcv1234")
MYSQL_DB = os.getenv("MYSQL_DB", "threat_intel")

def get_conn_pd():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DB,
        charset="utf8mb4",
        autocommit=True,
    )

# =========================================================
# 2) DB에서 데이터 가져오기
# =========================================================
@st.cache_data(ttl=60)
def load_risk_last_n_days(n_days: int = 30) -> pd.DataFrame:
    since = (date.today() - timedelta(days=n_days)).strftime("%Y-%m-%d")
    sql = """
    SELECT risk_date, doc_count, keyword_hits, score, recommended_threshold
    FROM threat_risk_daily
    WHERE risk_date >= %s
    ORDER BY risk_date ASC
    """
    conn = get_conn_pd()
    try:
        df = pd.read_sql_query(sql, conn, params=[since])
    finally:
        conn.close()
    return df

@st.cache_data(ttl=60)
def load_news_last_n_days(n_days: int = 2, limit: int = 50) -> pd.DataFrame:
    since_dt = (datetime.now() - timedelta(days=n_days)).strftime("%Y-%m-%d %H:%M:%S")
    sql = """
    SELECT id, source, keyword, url, title, published_at
    FROM threat_news_raw
    WHERE published_at >= %s
    ORDER BY published_at DESC
    LIMIT %s
    """
    conn = get_conn_pd()
    try:
        df = pd.read_sql_query(sql, conn, params=[since_dt, limit])
    finally:
        conn.close()
    return df

@st.cache_data(ttl=60)
def load_news_by_latest_date(limit: int = 50) -> pd.DataFrame:
    sql = """
    SELECT id, source, keyword, url, title, published_at
    FROM threat_news_raw
    WHERE DATE(published_at) = (SELECT DATE(MAX(published_at)) FROM threat_news_raw)
    ORDER BY published_at DESC
    LIMIT %s
    """
    conn = get_conn_pd()
    try:
        df = pd.read_sql_query(sql, conn, params=[limit])
    finally:
        conn.close()
    return df

@st.cache_data(ttl=60)
def load_keyword_top(n_days: int = 7, top_k: int = 10) -> pd.DataFrame:
    since_dt = (datetime.now() - timedelta(days=n_days)).strftime("%Y-%m-%d %H:%M:%S")
    sql = """
    SELECT keyword, COUNT(*) AS cnt
    FROM threat_news_raw
    WHERE published_at >= %s
    GROUP BY keyword
    ORDER BY cnt DESC
    LIMIT %s
    """
    conn = get_conn_pd()
    try:
        df = pd.read_sql_query(sql, conn, params=[since_dt, top_k])
    finally:
        conn.close()
    return df

@st.cache_data(ttl=60)
def load_recent_news(limit: int = 30) -> pd.DataFrame:
    sql = """
    SELECT source, keyword, url, title, published_at
    FROM threat_news_raw
    ORDER BY published_at DESC
    LIMIT %s
    """
    conn = get_conn_pd()
    try:
        df = pd.read_sql_query(sql, conn, params=[limit])
    finally:
        conn.close()
    return df

@st.cache_data(ttl=60)
def load_model_metrics_last_n_days(n_days: int = 30) -> pd.DataFrame:
    since = (date.today() - timedelta(days=n_days)).strftime("%Y-%m-%d")
    sql = """
    SELECT metric_date, model_name, method,
           precision_val, recall_val, f1_val, roc_auc_val, auprc_val
    FROM fds_model_metrics_daily
    WHERE metric_date >= %s
    ORDER BY metric_date ASC
    """
    conn = get_conn_pd()
    try:
        df = pd.read_sql_query(sql, conn, params=[since])
    finally:
        conn.close()
    return df

# =========================================================
# 3) 사이드바
# =========================================================
st.sidebar.markdown('<div class="sidebar-title">⚙️ 설정</div>', unsafe_allow_html=True)

n_days = st.sidebar.slider("위험도 그래프 기간(일)", 7, 30, 30, step=1)
kw_days = st.sidebar.slider("키워드 TOP 집계 기간(일)", 1, 30, 7, step=1)
news_days = st.sidebar.slider("뉴스 표시 기간(최근 N일)", 1, 30, 2, step=1)
news_limit = st.sidebar.slider("뉴스 표시 개수", 10, 200, 50, step=10)

refresh = st.sidebar.button("🔄 데이터 새로고침(캐시 무시)")
if refresh:
    st.cache_data.clear()
    st.rerun()

# =========================================================
# 4) 데이터 로드
# =========================================================
risk_df = load_risk_last_n_days(n_days)

news_df = load_news_last_n_days(news_days, news_limit)
news_mode = f"최근 {news_days}일"
if news_df.empty:
    news_df = load_news_by_latest_date(news_limit)
    news_mode = "DB 최신 날짜"

kw_df = load_keyword_top(kw_days, 10)
news_df = news_df.dropna(subset=["url"]).drop_duplicates(subset=["url"], keep="first")

metrics_df = load_model_metrics_last_n_days(n_days)

# =========================================================
# 5) 공통 유틸
# =========================================================
def safe_to_dt(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def safe_to_num(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fmt_raw(x, digits: int = 4):
    """반올림 없이 digits자리까지 '버림'해서 표시"""
    if pd.isna(x):
        return "-"

    try:
        d = Decimal(str(x))  # float 오차 줄이려고 str로 Decimal 변환
        q = Decimal("1." + "0" * digits)  # 예: digits=4 -> 1.0000
        d2 = d.quantize(q, rounding=ROUND_DOWN)  # ✅ 버림(반올림X)
        return format(d2, "f")  # 소수점 고정 표시
    except Exception:
        return str(x)


def fmt_int(x):
    if pd.isna(x):
        return "-"
    try:
        return str(int(x))
    except Exception:
        return str(x)

def method_is_raw(m: str) -> bool:
    """Raw/None/baseline 같은 항목은 화면에서 제외하기 위한 판별"""
    if not isinstance(m, str):
        return False
    m2 = m.strip().lower()
    return m2 in ("raw", "none", "no_aug", "no-aug", "baseline")

def canon_model_name(name: str) -> str:
    """모델명을 표준 표기(정렬/순서 고정 목적)"""
    if not isinstance(name, str):
        return str(name)
    n = name.strip().lower()
    if n in ("rf", "random_forest", "randomforest"):
        return "RandomForest"
    if n in ("tabtransformer", "tab_transformer", "tab-transformer"):
        return "TabTransformer"
    if n in ("tabnet", "tab_net", "tab-net"):
        return "TabNet"
    if n in ("autoencoder", "auto_encoder", "auto-encoder", "ae"):
        return "AutoEncoder"
    return name

MODEL_ORDER = ["RandomForest", "TabTransformer", "TabNet", "AutoEncoder"]

# =========================================================
# 6) 상단 네비(탭) + 헤더
# =========================================================
st.markdown(
    """
    <div class="topbar-wrap">
      <div class="badge">▦ Threat Intel FDS</div>
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["대시보드", "성능 분석", "뉴스 수집 현황"])

# =========================================================
# TAB 1) 대시보드
# =========================================================
with tabs[0]:
    st.markdown('<div class="title">Threat Intel 기반 FDS 대시보드</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">뉴스 크롤링(Scrapy) + 일일 위험도 점수(MySQL) + 시각화(Streamlit)</div>', unsafe_allow_html=True)

    if not risk_df.empty:
        risk_df = safe_to_dt(risk_df, "risk_date")
        risk_df = safe_to_num(risk_df, ["doc_count", "keyword_hits", "score", "recommended_threshold"])
        risk_df = risk_df.dropna(subset=["risk_date", "score"]).sort_values("risk_date")

    if not risk_df.empty:
        latest = risk_df.iloc[-1]
        kpi_html = f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">최근 위험도 점수(score)</div>
            <div class="kpi-value">{fmt_raw(latest.get("score"))}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">최근 키워드 히트(keyword_hits)</div>
            <div class="kpi-value">{fmt_int(latest.get("keyword_hits"))}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">최근 문서 수(doc_count)</div>
            <div class="kpi-value">{fmt_int(latest.get("doc_count"))}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">추천 임계치(recommended_threshold)</div>
            <div class="kpi-value">{fmt_raw(latest.get("recommended_threshold"))}</div>
          </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)
        st.markdown(
            f'<div class="okline">✅ 리스크 최신 날짜: {latest.get("risk_date").date()}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="kpi-grid">
              <div class="kpi-card"><div class="kpi-label">최근 위험도 점수(score)</div><div class="kpi-value">-</div></div>
              <div class="kpi-card"><div class="kpi-label">최근 키워드 히트(keyword_hits)</div><div class="kpi-value">-</div></div>
              <div class="kpi-card"><div class="kpi-label">최근 문서 수(doc_count)</div><div class="kpi-value">-</div></div>
              <div class="kpi-card"><div class="kpi-label">추천 임계치(recommended_threshold)</div><div class="kpi-value">-</div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="panel"><div class="panel-title">📈 위험도 추이</div></div>', unsafe_allow_html=True)
        if risk_df.empty:
            st.warning("threat_risk_daily 데이터가 없습니다.")
        else:
            y_min = float(risk_df["score"].min())
            y_max = float(risk_df["score"].max())
            if len(risk_df) == 1:
                y_min -= 5
                y_max += 5

            fig = px.line(risk_df, x="risk_date", y="score", markers=True)
            fig.update_traces(mode="lines+markers", marker=dict(size=9), line=dict(width=3))
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(range=[y_min, y_max]),
                xaxis_title="날짜",
                yaxis_title="위험도 점수(score)",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("원본 위험도 테이블 보기"):
                st.dataframe(risk_df, use_container_width=True)

    with right:
        st.markdown(
            f'<div class="panel"><div class="panel-title">📰 뉴스 ({news_mode} 기준)</div>'
            f'<div class="panel-sub">키워드 기반 수집 결과를 최근순으로 표시</div></div>',
            unsafe_allow_html=True
        )

        if news_df.empty:
            st.error("❌ 뉴스가 비어있습니다.")
        else:
            news_df = safe_to_dt(news_df, "published_at")
            mx = news_df["published_at"].max()
            if pd.notna(mx):
                st.markdown(f'<div class="okline">✅ 뉴스 최신 published_at: {mx}</div>', unsafe_allow_html=True)

            box = st.container(height=420)
            with box:
                for _, row in news_df.iterrows():
                    title = row.get("title") or "(제목없음)"
                    kw = row.get("keyword") or "-"
                    pub = row.get("published_at")
                    url = row.get("url") or ""
                    pub_txt = "-" if pd.isna(pub) else str(pub)

                    if url:
                        item = f"""
                        <div class="news-item">
                          <div style="font-weight:850; margin-bottom:4px;">
                            <a href="{url}" target="_blank" style="text-decoration:none; color: rgba(17,24,39,0.92);">
                              {title}
                            </a>
                          </div>
                          <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="news-tag">#{kw}</span>
                            <span class="news-time">{pub_txt}</span>
                          </div>
                        </div>
                        """
                    else:
                        item = f"""
                        <div class="news-item">
                          <div style="font-weight:850; margin-bottom:4px;">{title}</div>
                          <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="news-tag">#{kw}</span>
                            <span class="news-time">{pub_txt}</span>
                          </div>
                        </div>
                        """
                    st.markdown(item, unsafe_allow_html=True)

        st.write("")
        st.markdown(f'<div class="panel"><div class="panel-title">🔥 키워드 TOP (최근 {kw_days}일)</div></div>', unsafe_allow_html=True)
        if kw_df.empty:
            st.info("키워드 집계 데이터가 없습니다.")
        else:
            fig2 = px.bar(kw_df, x="keyword", y="cnt")
            fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# TAB 2) 성능 분석
# - 선택 지표 기준 챔피언(최고값) -> 그래프/표/카드 값一致
# - 모델 순서 고정(RandomForest, TabTransformer, TabNet, AutoEncoder)
# - Raw(method) 제외
# - 표에서 각 모델별 최고값 Bold+하이라이트
# =========================================================
with tabs[1]:
    st.markdown('<div class="title">AI Model Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">데이터 증강 기법과 알고리즘 조합을 통한 최적의 FDS 성능 탐색</div>', unsafe_allow_html=True)

    if metrics_df.empty:
        st.info("모델 지표 데이터가 없습니다. (fds_model_metrics_daily 테이블 확인)")
    else:
        metrics_df = safe_to_dt(metrics_df, "metric_date")
        metrics_df = safe_to_num(metrics_df, ["precision_val", "recall_val", "f1_val", "roc_auc_val", "auprc_val"])

        # 모델명 표준화(정렬용)
        metrics_df["model_name"] = metrics_df["model_name"].astype(str).apply(canon_model_name)

        # 최신 날짜 기준 표시(스크린샷 스타일)
        latest_date = metrics_df["metric_date"].max()
        df_latest = metrics_df[metrics_df["metric_date"] == latest_date].copy()
        if df_latest.empty:
            df_latest = metrics_df.copy()

        # ✅ Raw/baseline method 제거 (요청: method 안 쓸거면 빼고)
        df_latest = df_latest[~df_latest["method"].astype(str).apply(method_is_raw)].copy()

        # 지표 선택
        metric_map = {
            "AUPRC": "auprc_val",
            "Recall": "recall_val",
            "Precision": "precision_val",
            "F1-score": "f1_val",
        }
        picked_metric_label = st.selectbox("지표 선택", list(metric_map.keys()), index=0)
        picked_metric = metric_map[picked_metric_label]

        # ✅ 챔피언 = "선택 지표" 기준 최댓값 (그래프/카드/표 mismatch 해결)
        if df_latest[picked_metric].dropna().empty:
            st.warning(f"{picked_metric_label} 값이 비어있어 챔피언을 선택할 수 없습니다.")
        else:
            champ_idx = df_latest[picked_metric].idxmax()
            champ = df_latest.loc[champ_idx]

            champ_model = str(champ.get("model_name"))
            champ_method = str(champ.get("method"))
            champ_f1 = champ.get("f1_val")
            champ_p = champ.get("precision_val")
            champ_r = champ.get("recall_val")
            champ_auprc = champ.get("auprc_val")

            colA, colB = st.columns([1.65, 1])

            with colA:
                hero_html = f"""
                <div class="hero">
                  <div class="hero-top">
                    <div class="hero-pill">🏆 CURRENT CHAMPION (by {picked_metric_label})</div>
                  </div>
                  <div class="hero-title">{champ_method} + {champ_model}</div>
                  <div class="hero-desc">
                    현재 선택 지표(<b>{picked_metric_label}</b>) 기준으로 가장 높은 값을 기록한 조합입니다.
                    (그래프/표/카드 모두 동일 기준으로 계산)
                  </div>
                  <div class="hero-metrics">
                    <div class="mini"><div class="mini-label">F1-SCORE</div><div class="mini-value">{fmt_raw(champ_f1)}</div></div>
                    <div class="mini"><div class="mini-label">PRECISION</div><div class="mini-value">{fmt_raw(champ_p)}</div></div>
                    <div class="mini"><div class="mini-label">RECALL</div><div class="mini-value">{fmt_raw(champ_r)}</div></div>
                    <div class="mini"><div class="mini-label">AUPRC</div><div class="mini-value">{fmt_raw(champ_auprc)}</div></div>
                  </div>
                </div>
                """
                st.markdown(hero_html, unsafe_allow_html=True)

            with colB:
                st.markdown('<div class="panel"><div class="panel-title">🧭 PERFORMANCE BALANCE</div></div>', unsafe_allow_html=True)

                # 레이더는 숫자 필요 -> NaN이면 0 처리
                radar_labels = ["F1", "Precision", "Recall", "AUPRC"]
                radar_vals = [
                    float(champ_f1) if pd.notna(champ_f1) else 0.0,
                    float(champ_p) if pd.notna(champ_p) else 0.0,
                    float(champ_r) if pd.notna(champ_r) else 0.0,
                    float(champ_auprc) if pd.notna(champ_auprc) else 0.0,
                ]
                radar_labels2 = radar_labels + [radar_labels[0]]
                radar_vals2 = radar_vals + [radar_vals[0]]

                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=radar_vals2,
                        theta=radar_labels2,
                        fill="toself",
                        name="Champion",
                    )
                )
                fig_radar.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=20, b=10),
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
                    showlegend=False,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.write("")

            # -------------------------------
            # (C) 상세 비교(선택 지표) 그래프
            # - 모델 순서 고정 + 값/텍스트 반올림 제거
            # -------------------------------
            st.markdown('<div class="panel"><div class="panel-title">🔎 상세 비교(선택 지표)</div></div>', unsafe_allow_html=True)

            # 모델 순서 정렬(없는 모델은 자동 제외, 나머지는 뒤로)
            present_models = [m for m in MODEL_ORDER if m in df_latest["model_name"].unique()]
            extra_models = [m for m in df_latest["model_name"].unique() if m not in MODEL_ORDER]
            model_order_final = present_models + sorted(extra_models)

            # method 순서는 데이터 등장 순서를 유지 (필요하면 여기서 커스텀 가능)
            method_order_final = list(dict.fromkeys(df_latest["method"].astype(str).tolist()))

            df_plot = df_latest.copy()
            # text를 "DB 값 그대로"로 보여주기 위해 str()로 그대로 표시
            df_plot["_text"] = df_plot[picked_metric].apply(fmt_raw)

            fig_cmp = px.bar(
                df_plot,
                x="method",
                y=picked_metric,
                color="model_name",
                barmode="group",
                text="_text",
                category_orders={
                    "model_name": model_order_final,
                    "method": method_order_final,
                },
            )
            fig_cmp.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_cmp, use_container_width=True)

            st.write("")

            # -------------------------------
            # (D) Augmentation Impact / Model Robustness
            # - baseline = 선택 지표의 최소값
            # - advanced = 챔피언(선택 지표 max)
            # - improvement = % 계산 (표시는 소수 너무 길면 보기 불편해서 str로 출력)
            # -------------------------------
            colC, colD = st.columns([1.15, 1])

            baseline_score = df_latest[picked_metric].min()
            advanced_score = df_latest[picked_metric].max()

            if pd.notna(baseline_score) and float(baseline_score) != 0:
                improve = (float(advanced_score) - float(baseline_score)) / float(baseline_score) * 100.0
            else:
                improve = 0.0

            with colC:
                impact_html = f"""
                <div class="panel" style="padding:18px;">
                  <div class="panel-title">↗️ Augmentation Impact</div>
                  <div class="panel-sub">
                    선택 지표(<b>{picked_metric_label}</b>) 기준으로 최저 대비 챔피언 향상 폭을 요약합니다.
                  </div>
                  <div style="display:flex; gap:12px; margin-top:10px; flex-wrap:wrap;">
                    <div class="kpi-card" style="flex:1; min-width:220px; box-shadow:none;">
                      <div class="kpi-label">BASELINE (MIN)</div>
                      <div class="kpi-value">{fmt_raw(baseline_score)}</div>
                    </div>
                    <div class="kpi-card" style="flex:1; min-width:220px; box-shadow:none;">
                      <div class="kpi-label">ADVANCED (CHAMPION)</div>
                      <div class="kpi-value">{fmt_raw(advanced_score)}</div>
                    </div>
                  </div>
                  <div style="margin-top:10px; font-weight:850; color: rgba(17,24,39,0.88);">
                    ✅ 평균적으로 약 <span style="color:#4f46e5;">{fmt_raw(improve)}</span>% 개선 효과
                  </div>
                  <div style="margin-top:4px; color: rgba(31,41,55,0.68); font-size:.90rem;">
                    불균형 데이터 환경에서 생성/클러스터 기반 증강이 minority 패턴 학습에 유리할 수 있습니다.
                  </div>
                </div>
                """
                st.markdown(impact_html, unsafe_allow_html=True)

            with colD:
                robust_lines = []
                if "randomforest" in champ_model.lower():
                    robust_lines += [
                        "Ensemble 계열 모델은 정형 데이터 이상 탐지에 강점",
                        "증강 적용 시 Precision/Recall 트레이드오프가 안정적으로 유지되는 편",
                    ]
                if "tabtransformer" in champ_model.lower() or "tabnet" in champ_model.lower():
                    robust_lines += [
                        "Tab 계열은 피처 상호작용을 학습해 증강 없이도 비교적 견고",
                        "데이터가 늘어날수록 성능이 추가 개선될 가능성",
                    ]
                if "autoencoder" in champ_model.lower():
                    robust_lines += [
                        "AutoEncoder는 재구성 오류 기반 이상 탐지에 유리",
                        "임계치 튜닝/드리프트 감지 로직과 결합하면 운영 안정성 향상",
                    ]
                if not robust_lines:
                    robust_lines = [
                        "운영 환경에서는 Precision/Recall 균형을 목표로 임계치 튜닝을 병행 권장",
                        "데이터 드리프트 감지(분포 변화) 로직을 함께 두면 안정성 향상",
                    ]

                bullets = "".join([f"<li>{x}</li>" for x in robust_lines])

                robust_html = f"""
                <div class="panel" style="padding:18px;">
                  <div class="panel-title">🛡️ Model Robustness</div>
                  <div class="panel-sub">
                    챔피언 조합(<b>{champ_method} + {champ_model}</b>)의 운영 관점 해석 포인트
                  </div>
                  <ul style="margin-top:10px; color: rgba(31,41,55,0.80);">
                    {bullets}
                  </ul>
                </div>
                """
                st.markdown(robust_html, unsafe_allow_html=True)

            st.write("")

            # -------------------------------
            # (E) 매트릭스 표: Augmentation x Algorithm
            # - 모델 순서 고정
            # - 각 모델별 최고값 Bold/Highlight
            # - method 라벨(인덱스명) 숨김
            # -------------------------------
            st.markdown('<div class="panel"><div class="panel-title">📊 성능 비교 매트릭스 (Augmentation x Algorithm)</div></div>', unsafe_allow_html=True)

            pivot_df = df_latest.pivot_table(
                index="method",
                columns="model_name",
                values=picked_metric,
                aggfunc="max",
            )

            # 모델 컬럼 순서 강제
            cols_present = [c for c in MODEL_ORDER if c in pivot_df.columns]
            cols_extra = [c for c in pivot_df.columns if c not in MODEL_ORDER]
            pivot_df = pivot_df[cols_present + sorted(cols_extra)]

            # method 인덱스명 제거(표에서 'method' 글자 안 보이게)
            pivot_df.index.name = ""

            def highlight_max_per_col(s: pd.Series):
                """각 모델(컬럼)에서 max 값을 Bold + 배경색"""
                if s.dropna().empty:
                    return [""] * len(s)
                mx = s.max()
                styles = []
                for v in s:
                    if pd.notna(v) and v == mx:
                        styles.append("font-weight: 900; background-color: rgba(99,102,241,0.16);")
                    else:
                        styles.append("")
                return styles

            pivot_styled = pivot_df.style.apply(highlight_max_per_col, axis=0)

            with st.expander("표로 보기 (Augmentation x Algorithm)", expanded=True):
                # Streamlit은 Styler 표시 지원(버전 낮으면 일반 dataframe으로 fallback)
                try:
                    st.dataframe(pivot_styled, use_container_width=True)
                except Exception:
                    st.dataframe(pivot_df, use_container_width=True)

            st.write("")

            # -------------------------------
            # (F) OpenAI 자동 해설 (키 있으면)
            # -------------------------------
            st.markdown('<div class="panel"><div class="panel-title">🧠 OpenAI 기반 자동 해설</div></div>', unsafe_allow_html=True)

            @st.cache_data(ttl=3600)
            def gen_ai_explanation(model_name: str, method: str, f1, p, r, auprc, picked_metric_label: str) -> str:
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not api_key:
                    return (
                        "OPENAI_API_KEY가 설정되어 있지 않아 자동 해설을 생성하지 못했습니다.\n\n"
                        "✅ 설정 방법(예):\n"
                        "- Windows PowerShell: `$env:OPENAI_API_KEY=\"YOUR_KEY\"`\n"
                        "- macOS/Linux: `export OPENAI_API_KEY=\"YOUR_KEY\"`\n\n"
                        "키를 설정한 뒤 다시 실행하면, 챔피언 조합에 대한 해설이 자동 생성됩니다."
                    )

                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                except Exception:
                    return (
                        "OpenAI 라이브러리를 불러올 수 없습니다.\n"
                        "- `pip install openai`\n"
                        "- 또는 실행 환경에 openai 패키지 설치 여부를 확인하세요.\n"
                    )

                prompt = f"""
너는 '금융 이상거래 탐지(FDS)' 프로젝트 발표 자료를 쓰는 분석가야.
아래 결과를 보고, "왜 이 조합이 좋았는지"를 6~9줄로 간결하게 한국어로 설명해줘.
- 너무 과장하지 말고, 운영 관점(정탐/오탐, 임계치, 드리프트)도 1~2문장 포함
- 마지막 줄에 '다음 실험 제안'을 2개 bullet로 제시

[결과]
- Champion(선택 지표 기준): {method} + {model_name}
- F1: {fmt_raw(f1)}, Precision: {fmt_raw(p)}, Recall: {fmt_raw(r)}, AUPRC: {fmt_raw(auprc)}
- 현재 화면 기준 지표 선택: {picked_metric_label}
"""
                try:
                    res = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": "너는 데이터 기반으로 논리적으로 설명하는 한국어 기술 라이터다."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.35,
                    )
                    return res.choices[0].message.content.strip()
                except Exception as e:
                    return f"OpenAI 호출 중 오류가 발생했습니다: {e}"

            explain_text = gen_ai_explanation(
                champ_model, champ_method, champ_f1, champ_p, champ_r, champ_auprc, picked_metric_label
            )
            st.write(explain_text)

# =========================================================
# TAB 3) 뉴스 수집 현황
# =========================================================
with tabs[2]:
    st.markdown('<div class="title">최근 수집 뉴스(전체 최신순)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">DB(threat_news_raw)에 들어온 수집 결과를 테이블로 확인</div>', unsafe_allow_html=True)

    recent_df = load_recent_news(50)
    recent_df = safe_to_dt(recent_df, "published_at")

    try:
        st.dataframe(
            recent_df.reset_index(drop=True),
            use_container_width=True,
            column_config={"url": st.column_config.LinkColumn("URL", display_text="열기")},
        )
    except Exception:
        st.dataframe(recent_df.reset_index(drop=True), use_container_width=True)
