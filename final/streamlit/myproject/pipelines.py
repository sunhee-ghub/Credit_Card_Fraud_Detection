# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


class ThreatnewsPipeline:
    def process_item(self, item, spider):
        return item
# pipelines.py
import pymysql
from datetime import datetime

class MySQLStorePipeline:
    def __init__(self, host, port, user, password, db):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.conn = None
        self.cur = None

    @classmethod
    def from_crawler(cls, crawler):
        # settings.py에 적어둔 값 읽기
        return cls(
            host=crawler.settings.get("MYSQL_HOST"),
            port=crawler.settings.getint("MYSQL_PORT", 3306),
            user=crawler.settings.get("MYSQL_USER"),
            password=crawler.settings.get("MYSQL_PASSWORD"),
            db=crawler.settings.get("MYSQL_DB"),
        )

    def open_spider(self, spider):
        # 스파이더 시작할 때 DB 연결
        self.conn = pymysql.connect(
            host=self.host, port=self.port,
            user=self.user, password=self.password,
            db=self.db, charset="utf8mb4",
            autocommit=True
        )
        self.cur = self.conn.cursor()

    def close_spider(self, spider):
        # 스파이더 종료 시 DB 연결 닫기
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def process_item(self, item, spider):
        # published_at 문자열을 datetime으로 변환
        published_dt = None
        if item.get("published_at"):
            try:
                published_dt = datetime.strptime(item["published_at"], "%Y-%m-%d %H:%M")
            except:
                published_dt = None

        sql = """
        INSERT INTO threat_news_raw (source, keyword, url, title, published_at, body)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          keyword = VALUES(keyword),
          title = VALUES(title),
          published_at = VALUES(published_at),
          body = VALUES(body)
        """

        self.cur.execute(sql, (
            item.get("source", "boannews"),
            item.get("keyword"),
            item.get("url"),
            item.get("title"),
            published_dt,
            item.get("body"),
        ))

        return item
