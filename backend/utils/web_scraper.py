"""
web_scraper.py — Scrape websites recursively, clean HTML, extract structured content.
Supports depth-limited crawling with rate limiting.
"""
import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Set, Optional
from loguru import logger
import time
import re
from pathlib import Path
import json


class WebScraper:
    """
    Recursive website scraper with:
    - Depth-limited crawling
    - Content cleaning (removes nav, footer, ads)
    - Rate limiting
    - Metadata extraction (title, URL, headings)
    """

    # HTML tags to completely remove
    REMOVE_TAGS = [
        "script", "style", "nav", "footer", "header",
        "aside", "iframe", "noscript", "form", "button",
        "meta", "link", "advertisement"
    ]

    # CSS selectors likely to be noise
    NOISE_SELECTORS = [
        ".nav", ".navbar", ".footer", ".sidebar", ".advertisement",
        ".cookie-banner", ".popup", "#nav", "#footer", "#header",
        ".breadcrumb", ".pagination", ".social-share"
    ]

    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 50,
        delay: float = 1.0,
        cache_dir: str = "./data/scraped"
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay = delay
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.visited: Set[str] = set()
        self.pages: List[Dict[str, Any]] = []

    async def scrape(self, start_url: str) -> List[Dict[str, Any]]:
        """
        Scrape a website starting from start_url.

        Returns:
            List of dicts: {text, url, title, section, type}
        """
        self.visited.clear()
        self.pages.clear()
        base_domain = urlparse(start_url).netloc

        logger.info(f"Starting scrape: {start_url} (depth={self.max_depth}, max={self.max_pages})")

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
        ) as client:
            await self._crawl(client, start_url, base_domain, depth=0)

        logger.info(f"Scraped {len(self.pages)} pages from {start_url}")
        self._save_cache(start_url)
        return self.pages

    async def _crawl(
        self,
        client: httpx.AsyncClient,
        url: str,
        base_domain: str,
        depth: int
    ):
        """Recursively crawl pages."""
        if (
            url in self.visited
            or len(self.pages) >= self.max_pages
            or depth > self.max_depth
        ):
            return

        self.visited.add(url)

        try:
            await asyncio.sleep(self.delay)
            response = await client.get(url)

            if response.status_code != 200:
                return

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                return

            html = response.text
            page_data = self._parse_html(html, url)

            if page_data and len(page_data["text"]) > 100:
                self.pages.append(page_data)
                logger.debug(f"Scraped [{depth}]: {url}")

            # Extract links for further crawling
            if depth < self.max_depth:
                links = self._extract_links(html, url, base_domain)
                for link in links:
                    if len(self.pages) < self.max_pages:
                        await self._crawl(client, link, base_domain, depth + 1)

        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")

    def _parse_html(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Parse HTML into clean structured text with metadata."""
        soup = BeautifulSoup(html, "lxml")

        # Remove noise elements
        for tag in self.REMOVE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        for selector in self.NOISE_SELECTORS:
            for el in soup.select(selector):
                el.decompose()

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)

        # Extract main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.find("body")
        )

        if not main_content:
            return None

        # Extract headings for section metadata
        headings = []
        for h in main_content.find_all(["h1", "h2", "h3"]):
            text = h.get_text(strip=True)
            if text:
                headings.append(text)

        # Extract clean text with heading structure preserved
        text_blocks = []
        for element in main_content.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) > 20:
                if element.name in ["h1", "h2", "h3", "h4"]:
                    text_blocks.append(f"\n## {text}\n")
                else:
                    text_blocks.append(text)

        full_text = "\n".join(text_blocks)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # Clean excess newlines
        full_text = re.sub(r' {2,}', ' ', full_text)       # Clean excess spaces

        return {
            "text": full_text.strip(),
            "url": url,
            "title": title,
            "section": headings[0] if headings else title,
            "source": urlparse(url).netloc,
            "source_path": url,
            "page": None,
            "type": "web",
        }

    def _extract_links(self, html: str, base_url: str, base_domain: str) -> List[str]:
        """Extract internal links from HTML."""
        soup = BeautifulSoup(html, "lxml")
        links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            absolute_url = urljoin(base_url, href)
            parsed = urlparse(absolute_url)

            # Only follow links on the same domain
            if (
                parsed.netloc == base_domain
                and parsed.scheme in ["http", "https"]
                and "#" not in absolute_url
                and absolute_url not in self.visited
                and not any(ext in absolute_url.lower() for ext in [
                    ".pdf", ".jpg", ".png", ".gif", ".zip", ".doc"
                ])
            ):
                links.append(absolute_url)

        return list(set(links))

    def _save_cache(self, start_url: str):
        """Save scraped content to cache."""
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', start_url)[:50]
        cache_path = self.cache_dir / f"{safe_name}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.pages, f, indent=2, ensure_ascii=False)
        logger.info(f"Cache saved: {cache_path}")

    def load_cache(self, start_url: str) -> Optional[List[Dict]]:
        """Load previously scraped content from cache."""
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', start_url)[:50]
        cache_path = self.cache_dir / f"{safe_name}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None