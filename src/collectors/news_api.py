import requests
import pandas as pd
from src.utils.config import NEWS_KEY

def get_finance_news(query="chứng khoán", page_size=100):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "vi",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_KEY
    }

    r = requests.get(url, params=params).json()
    articles = r.get("articles", [])

    rows = [{
        "date": a["publishedAt"],
        "title": a["title"],
        "description": a["description"],
        "source": a["source"]["name"]
    } for a in articles]

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    return df
