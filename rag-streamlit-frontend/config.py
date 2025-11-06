#!/usr/bin/env python3
"""
Configuration file for Evidence Indicator RAG System Streamlit Frontend
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# Streamlit Configuration
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

# Sample Queries Organized by Category
SAMPLE_QUERIES_BY_CATEGORY = {
    "気象・梅雨": [
        "日本で梅雨がないのは北海道とどこか。",
        "梅雨とは何季の一種か?",
        "入梅は何の目安の時期か？",
        "梅雨明けの別名を何というか。",
        "シベリアから中国大陸にかけての広範囲を冷たく乾燥させる気団は？"
    ],
    "歴史・戦争": [
        "極東国際軍事裁判（東京裁判）で弁護人を務めたのは誰？",
        "「大東亜戦争」が閣議決定されたのはいつ？",
        "イギリスでは1918年初頭に導入されたのは？",
        "ジョゼフ・ジョフルが、フランス軍最高司令官の座を譲った相手は？",
        "イギリスにおける第一次世界大戦後の余剰女性はどれだけか？"
    ],
    "人物・思想家": [
        "マルクスが兵役不適格とされた理由は",
        "マルクスは何に不信感を抱いたか",
        "バウアーが大学で講義することは禁止された年は",
        "『インターナショナルにおける偽装的分裂』を採択したのはいつか",
        "松本清張の生まれは"
    ],
    "交通・鉄道": [
        "東海道新幹線で総列車本数の最大記録は何本か",
        "山陰地区発の特急は？",
        "平日朝の上りで約10分おきに三島駅から運行している東海道新幹線の種類は何か。",
        "スプリンクラーの設置は主に何処を中心とされているか？",
        "特別企画乗車券の有効期間は何日間か"
    ],
    "スポーツ・サッカー": [
        "ナビスコ杯では、グループリーグA組を4勝2敗の2位で通過したサッカーチームは何ですか。",
        "ベガルタ仙台の2002年の第2ステージの勝利数は？",
        "ベガルタ仙台がFC東京から完全移籍で獲得した選手は誰か",
        "J2得点王となったFWボルジェスは、期限付き移籍したチームは？",
        "仙台スタジアム の所在地は。"
    ],
    "地理・国家": [
        "タイの代表的な乗り物は。",
        "通称タイと呼ばれる国は",
        "タイ王国第31代首相は誰か？",
        "南部のマレー半島へはかつて、何朝が併合を目指して侵攻しましたか？",
        "東日本大震災が発生したのはいつ？"
    ],
    "科学・技術": [
        "古細菌の外観は何と似ているか。",
        "1674年に微生物を発見したのは誰か?",
        "最初に発売されたBDビデオソフトはDVDと同じ何をコーデックに採用せざるをえなかったか？",
        "-400以降の型式は",
        "一般的に使われている析出硬化系の母相の種類は具体的に何？"
    ],
    "政治・人物": [
        "アンが斬首されたのは何年か",
        "野村克也が監督の姿勢として一流と公言していたのは",
        "2008年6月29日のソフトバンク戦で球団史上最多20安打の猛攻で15点を奪い大勝した野球監督",
        "エリザベス1世の在位は何年か",
        "2010年オバマ政権は　何実験を行ったでしょうか"
    ],
    "文化・芸術": [
        "古今集と万葉集でより古いものは？",
        "松本清張の生まれは",
        "さくらが逝去した日",
        "木村の死去に際して清張は何を書いたか？",
        "日本刀の拵えなどに影響を与えたのは"
    ],
    "国際・外交": [
        "CPTPP は何のように加入交渉についての詳細な規定はしていない？",
        "フランスのどこから孫文が上海に帰国した？",
        "環太平洋パートナーシップ協定は当初何ヵ国か？",
        "革命活動の主要活動地域の一つとされるのはどこ？",
        "調印式会場のあるオークランドでは、約2万人の人々がTPPに反対するために抗議活動を行った協定は"
    ]
}

# Flatten all queries for backward compatibility
DEFAULT_QUERIES = []
for category_queries in SAMPLE_QUERIES_BY_CATEGORY.values():
    DEFAULT_QUERIES.extend(category_queries)

# UI Configuration
MAX_QUERY_HISTORY = 20
MAX_PERFORMANCE_METRICS = 100
DEFAULT_CHART_HEIGHT = 400

# Performance Thresholds
FAST_QUERY_THRESHOLD = 1.0  # seconds
SLOW_QUERY_THRESHOLD = 3.0  # seconds

# Colors for UI
COLORS = {
    'primary': '#1f77b4',
    'success': '#2ca02c', 
    'warning': '#ff7f0e',
    'error': '#d62728',
    'info': '#17a2b8'
} 