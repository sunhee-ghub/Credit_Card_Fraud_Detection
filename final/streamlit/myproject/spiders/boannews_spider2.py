import re   # (regular expression, 정규표현식) 모듈 (날짜 부분 추출용)
import scrapy
from urllib.parse import quote  # quote 함수 (URL 인코딩용)

class BoannewsSpider(scrapy.Spider):
    # =========================
    # 스파이더 기본 설정
    # =========================
    name = "boannews"                      # scrapy crawl boannews 로 실행할 때 쓰는 이름
    allowed_domains = ["boannews.com"]     # 이 도메인 밖으로는 크롤링하지 않도록 제한

    # =========================
    # 검색 키워드 목록
    # =========================
    keywords = ["카드", "사기", "유출"]     # 제목(title) 기준으로 검색할 키워드들

    # =========================
    # 전체뉴스 검색 URL 템플릿
    # - Page 파라미터로 페이지를 넘길 수 있음
    # - find={q} : EUC-KR 인코딩된 검색어가 들어갈 자리
    # =========================
    base_total_url = "https://www.boannews.com/search/news_total.asp?Page={page}&search=title&find={q}"

    # [추가] (전체 크롤링 동안) 이미 본 기사 링크를 저장할 Set
    # 이유:
    # - 보안뉴스는 Page 값을 올려도 마지막 페이지를 반복해서 보여주는 경우가 있음
    # - 그러면 hrefs는 계속 존재하지만(0이 아님) '새 기사'는 없어서 무한 루프가 발생
    # - 따라서 "이번 페이지에서 새 링크가 0개면 종료"를 하려면 전체 seen이 필요함
    seen_global = set()

    # [추가] 최대 페이지 제한 (원하면 숫자 조절 / 원치 않으면 None)
    # 이유:
    # - 사이트 구조 변화, 예외 상황에서도 무한 크롤링 방지용 안전장치
    # - 의미: 1페이지부터 MAX_PAGE까지만 수집하고 종료함
    MAX_PAGE = 300  # 예: 300 으로 두면 300페이지까지만 크롤링

    def start_requests(self):
        # =========================
        # 키워드별로 1페이지부터 시작
        # =========================
        for kw in self.keywords:
            # -------------------------
            # 한글 키워드를 EUC-KR로 URL 인코딩
            # (보안뉴스가 EUC-KR 기반 파라미터를 쓰는 경우가 많아서 이렇게 처리)
            # -------------------------
            q = quote(kw, encoding="euc-kr", errors="ignore")

            # -------------------------
            # 첫 페이지(1페이지) 검색 URL 생성
            # -------------------------
            url = self.base_total_url.format(page=1, q=q)

            # -------------------------
            # 검색 결과 1페이지 요청
            # meta로 keyword, q(인코딩된 값), page 정보를 넘겨서 다음 페이지로 이어가게 함
            # -------------------------
            yield scrapy.Request(
                url=url,                               # 요청할 URL
                callback=self.parse_total_page,         # 응답을 처리할 함수
                meta={"keyword": kw, "q": q, "page": 1},# 다음 요청에 필요한 정보들
                headers={"User-Agent": "Mozilla/5.0"},  # 간단한 UA 헤더
            )

    def parse_total_page(self, response):
        # =========================
        # 현재 페이지 메타 정보 꺼내기
        # =========================
        kw = response.meta["keyword"]          # 원본 키워드(한글)
        q = response.meta["q"]                 # EUC-KR 인코딩된 키워드
        page = response.meta["page"]           # 현재 페이지 번호

        # [추가] 최대 페이지 제한 체크
        # 이유:
        # - MAX_PAGE를 설정한 경우, 그 페이지까지만 크롤링하고 종료
        # - MAX_PAGE=None이면 이 기능은 꺼짐(기존 동작 유지)
        if self.MAX_PAGE is not None and page > self.MAX_PAGE:
            self.logger.info(f"[종료] keyword={kw} page={page} -> MAX_PAGE({self.MAX_PAGE}) 초과로 종료")
            return

        # =========================
        # 현재 페이지에서 기사 링크들 추출
        # - 검색 결과 페이지 HTML 안에 view.asp?idx= 형태로 기사 링크가 있음
        # =========================
        hrefs = response.css('a[href*="/media/view.asp?idx="]::attr(href)').getall()

        # =========================
        # (중요) 이번 페이지에서 기사 링크가 0개면
        # -> 더 이상 페이지가 없다고 판단하고 종료
        # =========================
        if not hrefs:
            self.logger.info(f"[종료] keyword={kw} page={page} 에서 기사 링크가 없어 종료합니다.")
            return

        # =========================
        # [수정] 중복 제거 범위를 "페이지 내" -> "전체(전역)"로 확장
        # 기존: seen=set()을 매 페이지마다 새로 만들었음 (페이지 내부 중복만 제거 가능)
        # 수정: self.seen_global로 관리 (전체 페이지 통틀어 이미 본 기사 링크를 제거)
        #
        # 이유:
        # - 마지막 페이지가 반복될 때 hrefs는 계속 나오지만, 전부 이미 본 링크라 '새 기사'가 0개가 됨
        # - 그 순간을 캐치해서 종료해야 무한 페이지네이션이 끊김
        # =========================
        new_count = 0  # [추가] 이번 페이지에서 새 기사 링크가 몇 개인지 카운트

        for href in hrefs:
            # [추가] 절대 URL로 통일해서 중복 판단 정확도 향상
            # 이유:
            # - href가 상대경로일 수 있으므로 urljoin으로 완전한 형태로 만들면 중복 판단이 안정적임
            abs_url = response.urljoin(href)

            if abs_url in self.seen_global:
                continue

            self.seen_global.add(abs_url)
            new_count += 1

            # -------------------------
            # 기사 상세 페이지로 이동
            # meta로 keyword 넘겨서 parse_article에서 사용
            # -------------------------
            yield response.follow(
                href,
                callback=self.parse_article,
                meta={"keyword": kw},
                headers={"User-Agent": "Mozilla/5.0"},
            )

        # [추가] 이번 페이지에서 '새 기사'가 0개면 종료
        # 이유:
        # - hrefs는 존재하지만 전부 이전에 수집한 링크인 경우(마지막 페이지 반복) 무한 루프 발생
        # - 새 링크가 0개면 사실상 더 이상 수집할 게 없다는 뜻이므로 페이지네이션 중단
        if new_count == 0:
            self.logger.info(f"[종료] keyword={kw} page={page} -> 새 기사 링크가 0개라 종료(마지막 페이지 반복 가능)")
            return

        # =========================
        # 다음 페이지로 이동 (페이지네이션)
        # - 현재 page가 1이면 next_page는 2
        # - 끝인지 여부는 위에서 "기사 링크가 없으면 종료" 로 처리
        # =========================
        next_page = page + 1

        # -------------------------
        # 다음 페이지 URL 생성
        # -------------------------
        next_url = self.base_total_url.format(page=next_page, q=q)

        # -------------------------
        # 다음 페이지 요청
        # meta에 page 값을 갱신해서 계속 넘기기
        # yield scrapy.Request -> 다음 페이지를 가져오는 작업을 예약
        # -------------------------
        yield scrapy.Request(
            url=next_url,
            callback=self.parse_total_page,
            meta={"keyword": kw, "q": q, "page": next_page},
            headers={"User-Agent": "Mozilla/5.0"},
        )

    def parse_article(self, response):
        # =========================
        # 상세 기사 페이지 파싱
        # =========================
        kw = response.meta["keyword"]  # 어떤 키워드에서 발견된 기사인지

        # =========================
        # 제목 추출
        # =========================
        title = response.css("div#news_title02 h1::text").get()
        title = title.strip() if title else None

        # =========================
        # 작성일(게시일) 추출
        # - div#news_util01 내부에 날짜/기자/조회수 등이 섞여있어서
        #   텍스트만 모은 다음 정규식으로 날짜만 뽑음
        #
        #  ".join(t.strip() for t in util_parts if t and t.strip()) : 텍스트 조각 한 줄로 합치기
        #
        # util_parts = [
        #  "\n",
        #  " 2026-01-27 13:20 ",
        #  "\t",
        #  "  기자명  ",
        #  "   ",
        #  "조회수 1234",
        #  "\n"
        # ]
        #  -----> "2026-01-27 13:20 기자명 조회수 1234"
        # =========================
        util_parts = response.css("div#news_util01::text").getall()
        util_text = " ".join(t.strip() for t in util_parts if t and t.strip())

        # -------------------------
        # YYYY-MM-DD HH:MM 형태의 날짜를 찾음
        # -------------------------
        m = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", util_text)
        published_at = m.group(1) if m else None

        # =========================
        # 본문 추출
        # - div#news_content 내부의 모든 텍스트를 긁고
        #   공백을 한 칸으로 정리
        # =========================
        body_parts = response.css("div#news_content *::text").getall()
        body = " ".join(t.strip() for t in body_parts if t and t.strip())
        body = re.sub(r"\s+", " ", body).strip()

        # =========================
        # 결과 반환
        # =========================
        yield {
            "source": "boannews",
            "keyword": kw,                 # 검색 키워드
            "url": response.url,           # ✅ 중복 방지용 핵심 키
            "title": title,                # 기사 제목
            "published_at": published_at,  # 게시 시간
            "body": body,                  # 본문 텍스트
        }