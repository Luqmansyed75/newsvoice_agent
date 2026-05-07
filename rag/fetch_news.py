import os
import re
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv


def detect_date_range(raw_query: str) -> int:
    """
    Scan the user's raw voice query for time-related keywords and return
    how many days back the NewsAPI should search.

    Examples:
        "today's IPL news"           → 1
        "yesterday's match result"   → 2
        "last week tech headlines"   → 7
        "this month's top stories"   → 30
        "give me AI news"            → 2  (default)
    """
    if not raw_query:
        return 2

    q = raw_query.lower()

    # Order matters — check the longest / most specific patterns first
    if re.search(r"last\s*month|past\s*month", q):
        return 30
    if re.search(r"this\s*month", q):
        return 30
    if re.search(r"last\s*week|past\s*week", q):
        return 7
    if re.search(r"this\s*week", q):
        return 7
    if "yesterday" in q:
        return 2
    if "today" in q or "latest" in q or "recent" in q:
        return 1

    # Default: last 2 days (covers "today" and "yesterday" naturally)
    return 2


def fetch_live_news(query=None, days_back: int = None):
    """
    Fetches the latest news from NewsAPI.org based on an optional query.
    Returns a single formatted string containing all the articles.

    Args:
        query:      Search keywords for the API.
        days_back:  How many days into the past to search.
                    If None, auto-detected from the raw query.
    """
    load_dotenv()
    
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key or api_key == "paste_your_new_key_here":
        print("WARNING: NEWS_API_KEY is not set in your .env file.")
        return None
        
    if query:
        url = "https://newsapi.org/v2/everything"
        # Dynamically compute the date window
        if days_back is None:
            days_back = detect_date_range(query)
        today = datetime.now()
        start_date = today - timedelta(days=days_back)
        print(f"[Date filter: searching the last {days_back} day(s)]")
        params = {
            "apiKey": api_key,
            "q": f"{query} India", # Focus on India news related to the query
            "pageSize": 10,
            "sortBy": "publishedAt",
            "from": start_date.strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d"),
        }
    else:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "apiKey": api_key,
            "country": "in", # Set to 'us' for US news, or 'in' for India
            "pageSize": 10   # Fetch the top 10 breaking news stories
        }
    
    print("[Connecting to NewsAPI...]")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception if the API key is invalid
        
        data = response.json()
        articles = data.get("articles", [])
        
        if not articles:
            print("No articles found!")
            return None
            
        formatted_text = "# Live Breaking News\n\n"
        
        for article in articles:
            title = article.get("title", "No Title")
            description = article.get("description", "")
            published_at = article.get("publishedAt", "")
            
            # Skip empty or deleted articles
            if title == "[Removed]":
                continue
            
            # Parse the raw timestamp into a clean, human-readable date
            date_label = ""
            if published_at:
                try:
                    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    date_label = dt.strftime("%B %d, %Y at %I:%M %p")
                except ValueError:
                    date_label = published_at

            formatted_text += f"## {title}\n"
            if date_label:
                formatted_text += f"[Published on: {date_label}]\n"
            if description:
                formatted_text += f"{description}\n"
            formatted_text += "\n"
            
        return formatted_text
        
    except Exception as e:
        print(f"Failed to fetch live news: {e}")
        return None

def update_news_file(query=None):
    """
    Fetches live news and overwrites the data/news.txt file with fresh content.
    """
    news_content = fetch_live_news(query=query)
    
    os.makedirs("data", exist_ok=True)
    if news_content:
        with open("data/news.txt", "w", encoding="utf-8") as f:
            f.write(news_content)
        print("[Successfully downloaded and saved live news!]")
        return True
    else:
        print("[No news found for this query. Clearing old data to prevent hallucinations.]")
        with open("data/news.txt", "w", encoding="utf-8") as f:
            f.write("No news articles found for your specific query.")
        return False

if __name__ == "__main__":
    # You can test this file directly by running `uv run rag/fetch_news.py`
    update_news_file()
