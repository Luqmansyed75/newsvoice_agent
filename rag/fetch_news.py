import os
import requests
from dotenv import load_dotenv

def fetch_live_news(query=None):
    """
    Fetches the latest news from NewsAPI.org based on an optional query.
    Returns a single formatted string containing all the articles.
    """
    load_dotenv()
    
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key or api_key == "paste_your_new_key_here":
        print("WARNING: NEWS_API_KEY is not set in your .env file.")
        return None
        
    if query:
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": api_key,
            "q": f"{query} India", # Focus on India news related to the query
            "pageSize": 10,
            "sortBy": "publishedAt"
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
            
            # Skip empty or deleted articles
            if title == "[Removed]":
                continue
                
            formatted_text += f"## {title}\n"
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
