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

# URL of the main sitemap
MAIN_SITEMAP_URL = "https://www.merckvetmanual.com/sitemap.xml"

# Headers to mimic a real browser visit
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# Get absolute path to current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output file to store extracted data
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "merck_knowledge.jsonl")

# File to store visited URLs for resumption
VISITED_URLS_FILE = os.path.join(SCRIPT_DIR, "visited_urls.json")

# Set to keep track of visited URLs to avoid duplication
visited_urls = set()

# Flag to control the crawling process
crawling = True

def signal_handler(sig, frame):
    """
    Handle termination signals to allow graceful shutdown.
    Saves the current state before exiting.
    """
    global crawling
    print("\nTermination signal received. Saving state and exiting gracefully...")
    save_visited_urls()
    sys.exit(0)

# Register the signal handler for graceful termination
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def fetch_sitemap_urls():
    """
    Fetch and parse the main XML sitemap to extract all article URLs.
    """
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
                    print(f"Failed to fetch sitemap: {sitemap_url} - Status code {response.status_code}")
                    continue

                # Try to decompress (if it's really gzip)
                try:
                    compressed = io.BytesIO(response.content)
                    decompressed = gzip.GzipFile(fileobj=compressed)
                    xml_content = decompressed.read()
                except (OSError, gzip.BadGzipFile):
                    xml_content = response.content

                soup = BeautifulSoup(xml_content, "lxml-xml")
                urls = [loc.get_text() for loc in soup.find_all("loc")]
                article_urls.extend(urls)

            except Exception as e:
                print(f"Error fetching sub-sitemap {sitemap_url}: {e}")

        return article_urls

    except Exception as e:
        print(f"Error fetching sitemaps: {e}")
        return []

def parse_article(url):
    """
    Parse an article page to extract structured content including title, section path, and full content block.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # Find main content container
        content_div = soup.find('div', {'data-testid': 'topic-main-content'})
        if not content_div:
            print(f"Skipped (no content section): {url}")
            return []

        # Extract title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"

        # Extract breadcrumb section hierarchy
        breadcrumb_ol = soup.find('ol', {'data-testid': 'breadcrumb-listWrap'})
        if breadcrumb_ol:
            breadcrumb_items = breadcrumb_ol.find_all('a', {'class': 'Breadcrumb_breadcrumbItemLink__zZn2u'})
            section = ' > '.join(item.get_text(strip=True) for item in breadcrumb_items)
        else:
            section = ""

        # Extract the full content block as plain text
        text = content_div.get_text(" ", strip=True)

        return [{
            "title": title,
            "section": section,
            "url": url,
            "content": text
        }]

    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return []

def save_to_jsonl(data, path=OUTPUT_FILE):
    """
    Save extracted data to a JSON Lines file.
    """
    with open(path, "a", encoding="utf-8") as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

def load_visited_urls(path=VISITED_URLS_FILE):
    """
    Load visited URLs from a file to support resumption and avoid duplication.
    """
    global visited_urls
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            visited_urls = set(json.load(f))

def save_visited_urls(path=VISITED_URLS_FILE):
    """
    Save visited URLs to a file to support resumption and avoid duplication.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(visited_urls), f)

def main():
    """
    Main function to orchestrate the crawling process.
    """
    global crawling

    # List of substrings to skip
    SKIP_PATTERNS = [
        "/authors/", "/videos/", "/calculators/", "/infographics/", "/quizzes/",
        "/case-study/", "/clinical-calculator/", "/multimedia/", "/resourcespages/",
        "/pages-with-widgets/","/news/"
    ]

    print("Loading previously visited URLs...")
    load_visited_urls()

    print("Fetching sitemap URLs...")
    urls = fetch_sitemap_urls()
    if not urls:
        print("No URLs found in the sitemaps. Exiting.")
        return

    print(f"Found {len(urls)} article URLs. Starting crawl...")

    for url in tqdm(urls):
        if not crawling:
            break
        if url in visited_urls:
            continue
        if any(skip in url for skip in SKIP_PATTERNS):
            print(f"Skipped by pattern: {url}")
            continue
        entries = parse_article(url)
        if entries:
            save_to_jsonl(entries)
        visited_urls.add(url)
        time.sleep(random.uniform(5, 6))  # Polite delay between requests

    print("Crawling completed. Saving visited URLs...")
    save_visited_urls()

if __name__ == "__main__":
    main()