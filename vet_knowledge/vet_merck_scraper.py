# pip install requests beautifulsoup4 tqdm

import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import urljoin
from tqdm import tqdm
import os

BASE_URL = "https://www.merckvetmanual.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
OUTPUT_FILE = "merck_knowledge.jsonl"

def get_all_article_links():
    """
    Crawl the site map to get all content article URLs.
    """
    sitemap_url = "https://www.merckvetmanual.com/mvm/sitemap"
    res = requests.get(sitemap_url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")
    
    links = []
    for a in soup.select(".sitemap li a"):
        href = a.get("href")
        if href and href.startswith("/"):
            full_url = urljoin(BASE_URL, href)
            links.append((full_url, a.text.strip()))
    return links

def parse_article(url, title):
    """
    Given an article URL, extract its paragraphs as structured knowledge.
    """
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            return []

        soup = BeautifulSoup(res.text, "html.parser")

        # Extract article breadcrumbs as 'section'
        breadcrumbs = soup.select("ul.breadcrumb li a")
        section = " > ".join(b.text.strip() for b in breadcrumbs[1:]) if breadcrumbs else ""

        # Extract article paragraphs
        content_div = soup.select_one("div.contentSection")
        if not content_div:
            return []

        entries = []
        for el in content_div.find_all(["p", "h2", "ul"]):
            text = el.get_text(" ", strip=True)
            if len(text) < 30:
                continue
            entries.append({
                "title": title,
                "section": section,
                "url": url,
                "paragraph": text
            })
        return entries

    except Exception as e:
        return []

def save_to_jsonl(data, path=OUTPUT_FILE):
    with open(path, "a", encoding="utf-8") as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

def load_existing_urls(path=OUTPUT_FILE):
    """
    Read existing file and collect already-saved URLs.
    """
    visited = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    visited.add(obj["url"])
                except:
                    continue
    return visited

def main():
    all_links = get_all_article_links()
    print(f"Found {len(all_links)} articles.")

    visited_urls = load_existing_urls()
    print(f"Skipping {len(visited_urls)} already crawled URLs.")

    for url, title in tqdm(all_links):
        if url in visited_urls:
            continue
        paragraphs = parse_article(url, title)
        if paragraphs:
            save_to_jsonl(paragraphs)
            visited_urls.add(url)
        time.sleep(random.uniform(0.5, 1.2))  # polite crawling

if __name__ == "__main__":
    main()

