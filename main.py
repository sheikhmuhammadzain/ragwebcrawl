#!/usr/bin/env python3
"""
Professional, configurable web scraper with:
- Retries and backoff (requests + urllib3 Retry)
- Rate limiting with jitter
- robots.txt respect (optional)
- Concurrency (ThreadPoolExecutor)
- Depth-limited crawl, same-domain option
- URL include/exclude regex filters
- JSONL or CSV output
- CLI with sensible defaults
- Streamlit UI and RAG-friendly dataset export

CLI Example:
  python me.py --urls https://example.com --same-domain --max-pages 200 --max-depth 3 --concurrency 8 --delay 0.5 --format json --output out.jsonl

UI Example:
  streamlit run me.py
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import threading
import time
import io
import hashlib
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urldefrag, urlsplit
import urllib.robotparser as robotparser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# Streamlit is optional for CLI usage; only required for the UI mode
try:
    import streamlit as st  # type: ignore
except Exception:  # ImportError or others in non-UI environments
    st = None  # Allows CLI usage without Streamlit installed


def normalize_url(url: str) -> str:
    """Strip whitespace and remove fragment (#...)."""
    u, _ = urldefrag((url or "").strip())
    return u


def get_netloc(url: str) -> str:
    return urlsplit(url).netloc.lower()


def is_http_like(url: str) -> bool:
    scheme = urlsplit(url).scheme.lower()
    return scheme in ("http", "https")


# -------------------------
# RAG helpers (chunking, cleaning)
# -------------------------
def normalize_whitespace(text: Optional[str]) -> str:
    """Collapse consecutive whitespace/newlines to single spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _smart_cut(text: str, start: int, size: int, lookahead: int = 120) -> int:
    """Cut near sentence/word boundary to keep chunks readable.
    Returns an end index >= start that attempts to end at a boundary.
    """
    n = len(text)
    end = min(start + size, n)
    if end >= n:
        return n
    window = text[end : min(end + lookahead, n)]
    # Prefer sentence enders, then whitespace
    for punct in [". ", "? ", "! ", "\n", " "]:
        idx = window.rfind(punct)
        if idx != -1:
            return end + idx + len(punct)
    return end


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    """Character-based chunking with soft sentence/word boundaries.
    Keeps small overlaps to preserve context for RAG.
    """
    text = normalize_whitespace(text)
    if not text:
        return []
    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size // 2))
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = _smart_cut(text, i, chunk_size)
        if end <= i:
            # Avoid infinite loop on pathological inputs
            end = min(i + chunk_size, n)
        out.append(text[i:end].strip())
        if end >= n:
            break
        i = max(0, end - chunk_overlap)
    return [c for c in out if c]


def build_rag_lines(record: Dict, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Dict]:
    """Transform a page record into RAG-friendly JSONL lines.
    Each line: {"id", "text", "metadata": { ... }}
    """
    base_meta = {
        "requested_url": record.get("requested_url"),
        "final_url": record.get("final_url"),
        "status": record.get("status"),
        "content_type": record.get("content_type"),
        "title": record.get("title"),
        "fetched_at": record.get("fetched_at"),
    }
    text = record.get("text") or ""
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    url_for_id = base_meta.get("final_url") or base_meta.get("requested_url") or ""
    ts_for_id = base_meta.get("fetched_at") or ""
    lines: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        raw_id = f"{url_for_id}|{ts_for_id}|{idx}"
        rid = hashlib.md5(raw_id.encode("utf-8")).hexdigest()
        lines.append({
            "id": rid,
            "text": chunk,
            "metadata": {**base_meta, "chunk_index": idx},
        })
    return lines


class OutputWriter:
    """Stream results to JSON Lines or CSV."""

    FIELDS = [
        "requested_url",
        "final_url",
        "status",
        "content_type",
        "title",
        "text",
        "num_links",
        "fetched_at",
        "error",
    ]

    def __init__(self, fmt: str, path: Optional[str]):
        self.fmt = (fmt or "json").lower()
        if self.fmt not in ("json", "csv"):
            raise ValueError("format must be 'json' or 'csv'")

        # Default output path by format if not provided
        if not path:
            path = "out.jsonl" if self.fmt == "json" else "out.csv"
        self.path = path

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        self._fh = None
        self._writer = None
        self._open()

    def _open(self):
        if self.fmt == "json":
            self._fh = open(self.path, "a", encoding="utf-8")
        else:
            file_exists = os.path.exists(self.path)
            is_empty = not file_exists or os.path.getsize(self.path) == 0
            self._fh = open(self.path, "a", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
            if is_empty:
                self._writer.writeheader()

    def write(self, record: Dict):
        if self.fmt == "json":
            # JSON Lines
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        else:
            row = {k: record.get(k) for k in self.FIELDS}
            # Convert lists to compact forms for CSV
            if isinstance(record.get("text"), str):
                # Keep as-is, may be large for some pages
                pass
            self._writer.writerow(row)
            self._fh.flush()

    def close(self):
        try:
            if self._fh:
                self._fh.close()
        finally:
            self._fh = None
            self._writer = None


class Scraper:
    def __init__(
        self,
        user_agent: str = "aifiesta-scraper/1.0",
        timeout: float = 15.0,
        delay: float = 0.5,
        concurrency: int = 8,
        max_pages: int = 100,
        max_depth: int = 2,
        same_domain: bool = True,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        respect_robots: bool = True,
        follow_redirects: bool = True,
        output_format: str = "json",
        output_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.delay = max(0.0, delay)
        self.concurrency = max(1, int(concurrency))
        self.max_pages = max(1, int(max_pages))
        self.max_depth = max(0, int(max_depth))
        self.same_domain = bool(same_domain)
        self.respect_robots = bool(respect_robots)
        self.follow_redirects = bool(follow_redirects)
        self.output_format = output_format
        self.output_path = output_path

        self.include_patterns = [re.compile(p, re.IGNORECASE) for p in (include or [])]
        self.exclude_patterns = [re.compile(p, re.IGNORECASE) for p in (exclude or [])]

        self.session = self._build_session()
        self.robots_cache: Dict[str, robotparser.RobotFileParser] = {}
        self._rl_lock = threading.Lock()
        self._last_request_ts = 0.0

        self.logger = logging.getLogger("scraper")
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=self.concurrency, pool_maxsize=self.concurrency)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def _robots_for(self, url: str) -> Optional[robotparser.RobotFileParser]:
        try:
            parts = urlsplit(url)
            base = f"{parts.scheme}://{parts.netloc}"
            robots_url = f"{base}/robots.txt"
            if robots_url not in self.robots_cache:
                rp = robotparser.RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                except Exception:
                    # If robots.txt cannot be read, treat as empty (allow)
                    pass
                self.robots_cache[robots_url] = rp
            return self.robots_cache[robots_url]
        except Exception:
            return None

    def _can_fetch(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        rp = self._robots_for(url)
        try:
            if rp is None:
                return True
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def _matches_filters(self, url: str) -> bool:
        if not is_http_like(url):
            return False
        if self.include_patterns:
            if not any(p.search(url) for p in self.include_patterns):
                return False
        if self.exclude_patterns and any(p.search(url) for p in self.exclude_patterns):
            return False
        return True

    def _is_allowed_domain(self, url: str, seed_domains: Set[str]) -> bool:
        if not self.same_domain:
            return True
        return get_netloc(url) in seed_domains

    def _rate_limit(self):
        if self.delay <= 0.0:
            return
        with self._rl_lock:
            now = time.monotonic()
            to_wait = self.delay - (now - self._last_request_ts)
            if to_wait > 0:
                # Add small jitter (20% of delay)
                jitter = random.uniform(0, self.delay * 0.2)
                time.sleep(to_wait + jitter)
            self._last_request_ts = time.monotonic()

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        out = []
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href:
                continue
            abs_url = normalize_url(urljoin(base_url, href))
            if is_http_like(abs_url):
                out.append(abs_url)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for u in out:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique

    def fetch(self, url: str) -> Tuple[Optional[Dict], List[str]]:
        self._rate_limit()
        requested_url = url
        try:
            resp = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=self.follow_redirects,
            )
            final_url = resp.url
            status = resp.status_code
            ctype = resp.headers.get("Content-Type", "") or ""

            is_html = "html" in ctype.lower()
            title = None
            text = None
            links: List[str] = []

            if is_html:
                # Decode via requests text property (charset aware)
                html = resp.text
                try:
                    soup = BeautifulSoup(html, "lxml")
                except Exception:
                    soup = BeautifulSoup(html, "html.parser")
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
                # Reasonable text extraction
                text = soup.get_text(separator=" ", strip=True)
                links = self._extract_links(html, final_url)

            record = {
                "requested_url": requested_url,
                "final_url": final_url,
                "status": status,
                "content_type": ctype,
                "title": title,
                "text": text,
                "num_links": len(links),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
            return record, links

        except requests.RequestException as e:
            self.logger.debug("Request error for %s: %s", requested_url, e, exc_info=True)
            record = {
                "requested_url": requested_url,
                "final_url": None,
                "status": None,
                "content_type": None,
                "title": None,
                "text": None,
                "num_links": 0,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
            return record, []
        except Exception as e:
            self.logger.debug("Unhandled error for %s: %s", requested_url, e, exc_info=True)
            record = {
                "requested_url": requested_url,
                "final_url": None,
                "status": None,
                "content_type": None,
                "title": None,
                "text": None,
                "num_links": 0,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "error": f"Unhandled: {e}",
            }
            return record, []

    def crawl(self, start_urls: Sequence[str], on_record: Optional[Callable[[Dict], None]] = None) -> int:
        start_urls = [normalize_url(u) for u in start_urls if u]
        start_urls = [u for u in start_urls if is_http_like(u)]
        if not start_urls:
            self.logger.error("No valid HTTP/HTTPS start URLs provided.")
            return 0

        seed_domains: Set[str] = {get_netloc(u) for u in start_urls}
        queue: deque[Tuple[str, int]] = deque((u, 0) for u in start_urls)
        visited: Set[str] = set()
        pages_written = 0

        writer = OutputWriter(self.output_format, self.output_path)
        future_depth: Dict = {}
        pending = set()

        try:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                while (queue or pending) and pages_written < self.max_pages:
                    # Fill up the pipeline
                    while (
                        queue
                        and len(pending) < self.concurrency
                        and (pages_written + len(pending) < self.max_pages)
                    ):
                        url, depth = queue.popleft()
                        if url in visited:
                            continue
                        if not self._matches_filters(url):
                            continue
                        if not self._is_allowed_domain(url, seed_domains):
                            continue
                        if self.respect_robots and not self._can_fetch(url):
                            continue

                        visited.add(url)
                        fut = pool.submit(self.fetch, url)
                        pending.add(fut)
                        future_depth[fut] = depth

                    if not pending:
                        break

                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        depth = future_depth.pop(fut, 0)
                        try:
                            record, links = fut.result()
                        except Exception as e:
                            self.logger.warning("Worker exception: %s", e, exc_info=True)
                            continue

                        if record:
                            # Persist
                            writer.write(record)
                            pages_written += 1
                            # Callback for UI progress/RAG
                            if on_record is not None:
                                try:
                                    on_record(record)
                                except Exception:
                                    # Keep crawling even if callback fails
                                    self.logger.debug("on_record callback failed", exc_info=True)
                            if pages_written % 10 == 0 or record.get("error"):
                                self.logger.info(
                                    "Pages: %d | Last: %s %s",
                                    pages_written,
                                    record.get("status"),
                                    record.get("requested_url"),
                                )

                        # Enqueue next depth
                        if links and depth + 1 <= self.max_depth:
                            for link in links:
                                if link not in visited and self._matches_filters(link) and self._is_allowed_domain(link, seed_domains):
                                    queue.append((link, depth + 1))

            self.logger.info("Finished. Total pages written: %d", pages_written)
            return pages_written
        finally:
            writer.close()


# -------------------------
# Streamlit UI (optional)
# -------------------------
def run_streamlit_app():
    if st is None:
        print("Streamlit is not installed. Install with: pip install streamlit", file=sys.stderr)
        return

    st.set_page_config(page_title="Zain Crawler • RAG Builder", layout="wide")
    st.title("Zain Web Crawler & RAG Dataset Builder")
    st.caption("Crawl websites professionally and export clean, chunked JSONL for RAG.")

    with st.sidebar:
        st.header("Configuration")
        urls_text = st.text_area(
            "Start URL(s)",
            placeholder="One per line, e.g.\nhttps://example.com\nhttps://docs.example.com",
            height=110,
        )

        st.subheader("Crawl Settings")
        same_domain = st.checkbox("Restrict to same domain(s)", value=True, help="Prevents cross-domain crawling")
        max_pages = st.slider("Max pages", min_value=1, max_value=5000, value=100, step=1)
        max_depth = st.slider("Max depth", min_value=0, max_value=10, value=2, step=1)
        concurrency = st.slider("Concurrency", min_value=1, max_value=64, value=8, step=1)
        delay = st.slider("Base delay (s)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        timeout = st.slider("Timeout (s)", min_value=5, max_value=120, value=15, step=5)
        follow_redirects = st.checkbox("Follow redirects", value=True)
        respect_robots = st.checkbox("Respect robots.txt", value=True)

        st.subheader("Filters")
        include_patterns = st.text_area("Include regex (optional)", placeholder=r"/docs|/blog", height=60)
        exclude_patterns = st.text_area("Exclude regex (optional)", placeholder=r"\.pdf$|\?", height=60)

        st.subheader("Headers")
        user_agent = st.text_input("User-Agent", value="zain-scraper/1.0")

        st.subheader("RAG Chunking")
        chunk_size = st.slider("Chunk size (chars)", min_value=300, max_value=4000, value=1200, step=50)
        chunk_overlap = st.slider("Chunk overlap (chars)", min_value=0, max_value=1000, value=150, step=10)

        st.markdown("---")
        run_btn = st.button("Start Crawl", type="primary", use_container_width=True)

    # Main area
    st.write("\n")
    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
    with col_a:
        st.metric("Max Pages", max_pages)
    with col_b:
        st.metric("Max Depth", max_depth)
    with col_c:
        st.metric("Concurrency", concurrency)
    with col_d:
        st.metric("Delay (s)", delay)

    progress = st.progress(0)
    status_placeholder = st.empty()
    preview_placeholder = st.container()
    download_placeholder = st.container()

    if run_btn:
        start_urls = [u.strip() for u in re.split(r"[\r\n,\s]+", urls_text or "") if u.strip()]
        if not start_urls:
            st.error("Please provide at least one valid HTTP/HTTPS URL.")
            st.stop()

        include_list = [s for s in re.split(r"[\r\n]+", include_patterns or "") if s.strip()]
        exclude_list = [s for s in re.split(r"[\r\n]+", exclude_patterns or "") if s.strip()]

        records: List[Dict] = []
        rag_lines: List[Dict] = []

        def on_record_cb(rec: Dict):
            records.append(rec)
            rag_lines.extend(build_rag_lines(rec, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
            # Update progress UI
            pct = min(1.0, len(records) / float(max_pages))
            progress.progress(int(pct * 100))
            status_placeholder.info(f"Crawled {len(records)} / {max_pages} pages")

        scraper = Scraper(
            user_agent=user_agent,
            timeout=float(timeout),
            delay=float(delay),
            concurrency=int(concurrency),
            max_pages=int(max_pages),
            max_depth=int(max_depth),
            same_domain=bool(same_domain),
            include=include_list,
            exclude=exclude_list,
            respect_robots=bool(respect_robots),
            follow_redirects=bool(follow_redirects),
            output_format="json",
            output_path="out.jsonl",
            verbose=False,
        )

        t0 = time.time()
        total = scraper.crawl(start_urls, on_record=on_record_cb)
        dt = time.time() - t0
        progress.progress(100)

        status_placeholder.success(f"Finished. Pages written: {total} in {dt:.1f}s. RAG chunks: {len(rag_lines)}")

        # Preview
        with preview_placeholder:
            st.subheader("Preview: RAG JSONL (first 3 lines)")
            preview_lines = [json.dumps(x, ensure_ascii=False) for x in rag_lines[:3]]
            if preview_lines:
                st.code("\n".join(preview_lines), language="json")
            else:
                st.info("No preview available.")

        # Downloads
        raw_buf = io.StringIO()
        for rec in records:
            raw_buf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        raw_bytes = raw_buf.getvalue().encode("utf-8")

        rag_buf = io.StringIO()
        for line in rag_lines:
            rag_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
        rag_bytes = rag_buf.getvalue().encode("utf-8")

        with download_placeholder:
            st.subheader("Download")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download raw pages (JSONL)",
                    data=raw_bytes,
                    file_name="out.jsonl",
                    mime="application/json",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Download RAG dataset (JSONL)",
                    data=rag_bytes,
                    file_name="out_rag.jsonl",
                    mime="application/json",
                    use_container_width=True,
                )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Professional web scraper")
    p.add_argument("--urls", "-u", nargs="+", required=True, help="Start URL(s)")
    p.add_argument("--same-domain", dest="same_domain", action="store_true", help="Restrict crawl to the same domain(s) as start URLs")
    p.add_argument("--no-same-domain", dest="same_domain", action="store_false", help="Allow cross-domain crawling")
    p.set_defaults(same_domain=True)

    p.add_argument("--max-pages", type=int, default=100, help="Maximum number of pages to write")
    p.add_argument("--max-depth", type=int, default=2, help="Maximum link depth to follow")
    p.add_argument("--concurrency", type=int, default=8, help="Number of concurrent requests")
    p.add_argument("--delay", type=float, default=0.5, help="Base delay between requests (seconds)")
    p.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout (seconds)")

    p.add_argument("--include", nargs="*", default=[], help="Regex patterns URLs must match")
    p.add_argument("--exclude", nargs="*", default=[], help="Regex patterns URLs must NOT match")

    p.add_argument("--user-agent", default="aifiesta-scraper/1.0", help="User-Agent header")
    p.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    p.add_argument("--output", help="Output file path (default: out.jsonl for json or out.csv for csv)")
    p.add_argument("--respect-robots", dest="respect_robots", action="store_true", help="Respect robots.txt rules")
    p.add_argument("--ignore-robots", dest="respect_robots", action="store_false", help="Ignore robots.txt")
    p.set_defaults(respect_robots=True)

    p.add_argument("--no-redirects", dest="follow_redirects", action="store_false", help="Do not follow redirects")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # Default output path based on format if not provided
    output_path = args.output or ("out.jsonl" if args.format == "json" else "out.csv")

    scraper = Scraper(
        user_agent=args.user_agent,
        timeout=args.timeout,
        delay=args.delay,
        concurrency=args.concurrency,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        same_domain=args.same_domain,
        include=args.include,
        exclude=args.exclude,
        respect_robots=args.respect_robots,
        follow_redirects=args.follow_redirects,
        output_format=args.format,
        output_path=output_path,
        verbose=args.verbose,
    )
    try:
        total = scraper.crawl(args.urls)
        return 0 if total > 0 else 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    # Heuristic: if CLI flags for URLs are provided, run CLI. Otherwise try UI.
    has_cli_urls = any(a in ("--urls", "-u") for a in sys.argv[1:])
    if has_cli_urls:
        sys.exit(main())
    else:
        # Prefer Streamlit UI when invoked without CLI args
        try:
            run_streamlit_app()
        except SystemExit:
            # Allow Streamlit to manage its own exit signals
            pass