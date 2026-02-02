# compute_rolling_risk.py
# =========================
# ✅ MySQL에 저장된 뉴스들을 읽어서
# ✅ 최근 N일(rolling window) 뉴스 기반 위험도 점수(0~100)를 계산해
# ✅ threat_risk_daily 테이블에 저장하는 스크립트
# =========================

import re
import math
import pymysql
from collections import Counter, defaultdict
from datetime import datetime, date, timedelta
from konlpy.tag import Okt

# =========================
# 1) MySQL 접속 정보
# =========================
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "zxcv1234"
MYSQL_DB = "threat_intel"

# =========================
# 2) 위험 키워드 사전(예시)
# =========================
RISK_LEXICON = {
    "피싱": 3, "스미싱": 3, "보이스피싱": 4,
    "해킹": 3, "침해": 3,
    "유출": 4, "탈취": 4,
    "악성": 3, "악성코드": 4, "랜섬웨어": 4,
    "계정": 2, "가짜앱": 3,
    "카드복제": 4, "복제": 3,
    "정보": 1, "개인정보": 4,
    "위조": 3, "불법": 2,
    "사기": 3, "금융사기": 4,
}

# =========================
# 3) 점수 스케일링 (포화 방지: 로그 스케일)
# =========================
def to_score(weighted_hits: float, doc_count: int) -> int:
    if doc_count <= 0:
        return 0

    raw = weighted_hits / doc_count  # 기사 1개당 위험강도

    # ✅ raw가 커져도 점수가 천천히 증가하도록 log 스케일
    # denom(50)은 튜닝 포인트: 데이터 보면서 30~200 사이 조정
    score = int(100 * math.log1p(raw) / math.log1p(50))
    return max(0, min(100, score))

# =========================
# 4) 형태소 분석 + 위험 키워드 카운트 (최근성 반영 가중치 포함)
# =========================
def analyze_rows(rows, window_end_dt: datetime, half_life_days: float = 3.0):
    """
    rows: [(body, published_at), ...]
    half_life_days: 최근 뉴스가 얼마나 더 중요하게 반영될지(3일이면 3일 지나면 영향 1/2)
    """
    okt = Okt()

    # half-life 기반 감쇠 계수
    lam = math.log(2) / half_life_days

    total_weighted_hits = 0.0
    keyword_counter = defaultdict(float)

    for body, pub in rows:
        if not body:
            continue

        # ✅ 최신 기사일수록 weight가 1에 가까움
        delta_days = (window_end_dt - pub).total_seconds() / (3600 * 24)
        if delta_days < 0:
            delta_days = 0

        time_weight = math.exp(-lam * delta_days)

        merged = re.sub(r"\s+", " ", str(body)).strip()
        if not merged:
            continue

        nouns = okt.nouns(merged)
        nouns = [n for n in nouns if len(n) >= 2]
        freq = Counter(nouns)

        # ✅ 위험 키워드 가중합(기사별) + 최근성 가중치(time_weight) 적용
        for k, w in RISK_LEXICON.items():
            c = freq.get(k, 0)
            if c > 0:
                keyword_counter[k] += c * time_weight   # 빈도 집계(최근성 반영)
                total_weighted_hits += (c * w) * time_weight  # 점수용(가중치+최근성)

    # 상위 키워드 문자열화 (저장용)
    top_items = sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    top_keywords_str = ", ".join([f"{k}:{v:.1f}" for k, v in top_items])

    return total_weighted_hits, top_keywords_str

# =========================
# 5) 특정 날짜를 기준으로 최근 window_days 뉴스로 점수 계산 -> 저장
# =========================
def compute_and_save_for_date(target_date: date, window_days: int, conn):
    with conn.cursor() as cur:
        # ✅ window_days 범위: target_date 포함 과거 window_days일
        start_date = target_date - timedelta(days=window_days - 1)

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(target_date, datetime.max.time())

        sql = """
        SELECT body, published_at
        FROM threat_news_raw
        WHERE published_at BETWEEN %s AND %s
          AND body IS NOT NULL
          AND body <> ''
        """
        cur.execute(sql, (start_dt, end_dt))
        rows = cur.fetchall()

        doc_count = len(rows)

        weighted_hits, top_keywords_str = analyze_rows(
            rows,
            window_end_dt=end_dt,
            half_life_days=3.0  # ✅ 최근성 강도(튜닝 가능)
        )

        score = to_score(weighted_hits, doc_count)

        # ✅ 임계치 추천: score 높으면 조금 낮추되, 너무 과격하지 않게
        base_threshold = 0.030
        recommended_threshold = max(0.005, base_threshold * (1.0 - (score / 300.0)))

        upsert = """
        INSERT INTO threat_risk_daily
            (risk_date, doc_count, keyword_hits, score, top_keywords, recommended_threshold)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            doc_count = VALUES(doc_count),
            keyword_hits = VALUES(keyword_hits),
            score = VALUES(score),
            top_keywords = VALUES(top_keywords),
            recommended_threshold = VALUES(recommended_threshold)
        """
        cur.execute(
            upsert,
            (target_date, doc_count, weighted_hits, score, top_keywords_str, recommended_threshold)
        )

        return doc_count, weighted_hits, score, recommended_threshold

# =========================
# 6) 최근 lookback_days일치 "그래프용 점수"를 한 번에 채우기
# =========================
def run(lookback_days: int = 30, window_days: int = 3):
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        db=MYSQL_DB, charset="utf8mb4"
    )

    try:
        today = date.today()

        # ✅ 최근 lookback_days 날짜를 하나씩 계산해서 저장
        for i in range(lookback_days - 1, -1, -1):
            d = today - timedelta(days=i)
            doc_count, hits, score, thr = compute_and_save_for_date(d, window_days, conn)
            conn.commit()
            print(f"[OK] {d} window={window_days}d docs={doc_count} hits={hits:.1f} score={score} thr={thr:.6f}")
    finally:
        conn.close()

if __name__ == "__main__":
    run(lookback_days=30, window_days=3)
