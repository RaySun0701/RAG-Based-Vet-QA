import requests
import gzip
import io
import json
import time
import random
import os
import signal
import sys
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm

# Base URL of the Merck Veterinary Manual website
BASE_URL = "https://www.merckvetmanual.com"
MAIN_SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# === Path Configuration ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "vet_knowledge", "merck_knowledge.jsonl")
VISITED_URLS_FILE = os.path.join(PROJECT_ROOT, "vet_knowledge", "visited_urls.json")

visited_urls = set()
crawling = True

def signal_handler(sig, frame):
    global crawling
    print("\nTermination signal received. Saving state and exiting gracefully...")
    save_visited_urls()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def fetch_sitemap_urls():
    try:
        response = requests.get(MAIN_SITEMAP_URL, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch main sitemap: Status code {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, "lxml-xml")
        sitemap_urls = [loc.get_text() for loc in soup.find_all("loc")]
        article_urls = []

        for sitemap_url in sitemap_urls:
            try:
                response = requests.get(sitemap_url, headers=HEADERS, stream=True)
                if response.status_code != 200:
                    print(f"Failed to fetch sitemap: {sitemap_url}")
                    continue

                try:
                    xml_content = gzip.GzipFile(fileobj=io.BytesIO(response.content)).read()
                except (OSError, gzip.BadGzipFile):
                    xml_content = response.content

                soup = BeautifulSoup(xml_content, "lxml-xml")
                article_urls.extend(loc.get_text() for loc in soup.find_all("loc"))
            except Exception as e:
                print(f"Error fetching {sitemap_url}: {e}")

        return article_urls
    except Exception as e:
        print(f"Error fetching sitemaps: {e}")
        return []

def parse_article(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: Status {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find('div', {'data-testid': 'topic-main-content'})
        if not content_div:
            print(f"Skipped (no content section): {url}")
            return []

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"

        breadcrumb_ol = soup.find('ol', {'data-testid': 'breadcrumb-listWrap'})
        section = ' > '.join(
            a.get_text(strip=True) for a in breadcrumb_ol.find_all('a', class_='Breadcrumb_breadcrumbItemLink__zZn2u')
        ) if breadcrumb_ol else ""

        paragraphs = []
        for p in content_div.find_all('p', {'data-testid': 'topicPara'}):
            text = ' '.join(span.get_text(strip=True) for span in p.find_all('span', {'data-testid': 'topicText'}))
            if len(text) > 30:
                paragraphs.append(text)

        if not paragraphs:
            return []

        return [{
            "title": title,
            "section": section,
            "url": url,
            "paragraphs": paragraphs
        }]
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return []

def save_to_jsonl(data, path=OUTPUT_FILE):
    with open(path, "a", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

def load_visited_urls(path=VISITED_URLS_FILE):
    global visited_urls
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            visited_urls = set(json.load(f))

def save_visited_urls(path=VISITED_URLS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(visited_urls), f)

def main():
    global crawling

    SKIP_PATTERNS = [
        "/authors/", "/videos/", "/calculators/", "/infographics/", "/quizzes/",
        "/case-study/", "/clinical-calculator/", "/multimedia/", "/resourcespages/",
        "/pages-with-widgets/", "/news/"
    ]

    print("Loading previously visited URLs...")
    load_visited_urls()

    print("Fetching sitemap URLs...")
    urls = fetch_sitemap_urls()
    if not urls:
        print("No URLs found. Exiting.")
        return

    print(f"Found {len(urls)} URLs. Starting crawl...")

    for url in tqdm(urls):
        if not crawling or url in visited_urls or any(skip in url for skip in SKIP_PATTERNS):
            continue
        entries = parse_article(url)
        if entries:
            save_to_jsonl(entries)
        visited_urls.add(url)
        time.sleep(random.uniform(5, 6))

    print("Crawling completed. Saving visited URLs...")
    save_visited_urls()

if __name__ == "__main__":
    main()

