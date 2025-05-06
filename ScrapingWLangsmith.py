import os
import nest_asyncio
from dotenv import load_dotenv, find_dotenv
from bs4 import BeautifulSoup
import requests
import logging

from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
from langchain_together import Together
from langchain.agents.agent_types import AgentType
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

# Enable debugging
logging.basicConfig(level=logging.DEBUG)

# Fix asyncio for Jupyter or nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv(find_dotenv())

# Set LangSmith tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")

# Proxy support
proxy = os.getenv("PROXY")
if proxy and not proxy.startswith("http"):
    proxy = "http://" + proxy
if proxy:
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy

# Scraping Tool
@tool
def search_sciencedirect_bipolar(query: str) -> str:
    """
    Uses Selenium to search ScienceDirect for articles about bipolar disorder and returns the most cited one.
    """
    url = f"https://www.sciencedirect.com/search?qs={query.replace(' ', '%20')}"

    # Set up headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.select("article.result-item-content")

        results = []
        for art in articles[:10]:
            title_tag = art.select_one("h2 a")
            title = title_tag.text.strip() if title_tag else "No Title"
            link = "https://www.sciencedirect.com" + title_tag["href"] if title_tag else ""
            citation_tag = art.select_one("span.CitationCount")
            citations = citation_tag.text.strip() if citation_tag else "0"

            try:
                citations_int = int(citations)
            except ValueError:
                citations_int = 0

            results.append((title, link, citations_int))

        if not results:
            return "No articles found."

        most_cited = max(results, key=lambda x: x[2])
        return f"{most_cited[0]} ({most_cited[2]} citations): {most_cited[1]}"

    except Exception as e:
        return f"Error during scraping: {e}"

    finally:
        driver.quit()

# LLM Setup
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=512,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

# Agent
tools = [search_sciencedirect_bipolar]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run
response = agent.invoke({
    "input": "Find the most cited article about bipolar disorder on ScienceDirect. Include link and citation count."
})

print(response)
