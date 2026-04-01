from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

from grounding import extract_focus_phrase
from presets import is_medical_query
from retrieval import STOPWORDS, chunk_text_by_words, tokenize_retrieval_text
from utils import ensure_dir


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
MEDICAL_TRUST_TIERS = {
    "medlineplus.gov": 9.5,
    "nih.gov": 9.0,
    "cdc.gov": 9.0,
    "who.int": 8.8,
    "nhs.uk": 8.7,
    "mayoclinic.org": 8.6,
    "merckmanuals.com": 8.4,
    "msdmanuals.com": 8.4,
    "clevelandclinic.org": 8.3,
    "pubmed.ncbi.nlm.nih.gov": 8.1,
    "hopkinsmedicine.org": 7.8,
    "mountsinai.org": 7.8,
    "mskcc.org": 7.7,
    "upmc.com": 7.4,
    "tgh.org": 7.1,
    "medscape.com": 6.9,
    "webmd.com": 6.4,
    "healthline.com": 6.0,
    "medicalnewstoday.com": 5.8,
}
GENERAL_TRUST_TIERS = {
    "wikipedia.org": 4.5,
    "britannica.com": 5.0,
    "nationalgeographic.com": 4.8,
    "smithsonianmag.com": 4.7,
    "nasa.gov": 6.0,
    "noaa.gov": 6.0,
}
BAD_URL_PATTERNS = (
    "/search",
    "/login",
    "/signin",
    "/privacy",
    "/terms",
    "/account",
)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
FOLLOW_UP_TOPIC_TERMS = {
    "symptom",
    "symptoms",
    "cause",
    "causes",
    "treat",
    "treatment",
    "diagnosis",
    "prevent",
    "prevention",
    "complication",
    "complications",
    "prognosis",
    "types",
}
SEARCH_NOISE_TERMS = {
    "common",
    "current",
    "today",
    "latest",
    "recent",
    "effect",
    "effects",
    "impact",
    "impacts",
}


def _domain_weight(domain: str, medical: bool) -> float:
    if medical:
        for candidate, weight in MEDICAL_TRUST_TIERS.items():
            if domain.endswith(candidate):
                return weight
    for candidate, weight in GENERAL_TRUST_TIERS.items():
        if domain.endswith(candidate):
            return weight
    return 0.0


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    domain: str
    rank: int


@dataclass
class WebPage:
    title: str
    url: str
    text: str
    domain: str


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        params = parse_qs(parsed.query)
        if "uddg" in params and params["uddg"]:
            return unquote(params["uddg"][0])
    if "bing.com" in parsed.netloc and parsed.path.startswith("/ck/a"):
        params = parse_qs(parsed.query)
        encoded = params.get("u", [""])[0]
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        if encoded:
            padded = encoded + "=" * ((4 - len(encoded) % 4) % 4)
            try:
                decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                decoded = ""
            if decoded.startswith("http"):
                return decoded
    return url


def _url_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _cache_key(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}.json"


def _http_get(url: str, timeout_seconds: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _query_variants(query: str) -> List[str]:
    variants = []
    for candidate in (extract_focus_phrase(query), query.strip()):
        normalized = " ".join(candidate.split()).strip()
        if normalized and normalized not in variants:
            variants.append(normalized)
    lowered = " ".join(query.lower().split()).strip()
    effect_match = re.search(r"what does (.+?) do to (?:your |the )?([a-z][a-z\s]{1,40})$", lowered)
    if effect_match:
        subject = effect_match.group(1).strip()
        target = effect_match.group(2).strip()
        for candidate in (
            f"{subject} {target} effects",
            f"{subject} and {target} health",
            f"can {subject} affect {target}",
            f"{subject} impact on {target}",
        ):
            normalized = " ".join(candidate.split()).strip()
            if normalized and normalized not in variants:
                variants.append(normalized)
    affect_match = re.search(r"(.+?) affect(?:s)?(?: on)? (?:your |the )?([a-z][a-z\s]{1,40})$", lowered)
    if affect_match:
        subject = affect_match.group(1).strip()
        target = affect_match.group(2).strip()
        for candidate in (
            f"{subject} {target} effects",
            f"can {subject} affect {target}",
            f"{subject} impact on {target}",
            f"{subject} and {target} health",
        ):
            normalized = " ".join(candidate.split()).strip()
            if normalized and normalized not in variants:
                variants.append(normalized)
    return variants


def _search_variants(query: str) -> List[str]:
    variants = _query_variants(query)
    if is_medical_query(query):
        focus = extract_focus_phrase(query) or query.strip()
        for domain in ("medlineplus.gov", "mayoclinic.org", "nhs.uk", "cdc.gov"):
            candidate = f"site:{domain} {focus}".strip()
            if candidate not in variants:
                variants.append(candidate)
    return variants


def _anchor_term_coverage(terms: set[str], haystack_terms: set[str]) -> float:
    if not terms:
        return 1.0
    matched = terms.intersection(haystack_terms)
    return len(matched) / max(1, len(terms))


class WebResearchClient:
    def __init__(
        self,
        cache_dir: str | Path,
        timeout_seconds: float = 6.0,
        search_cache_ttl_seconds: int = 3600,
        page_cache_ttl_seconds: int = 86400,
    ):
        self.cache_dir = ensure_dir(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.search_cache_ttl_seconds = search_cache_ttl_seconds
        self.page_cache_ttl_seconds = page_cache_ttl_seconds

    def _read_cache(self, key: str, ttl_seconds: int) -> Dict[str, object] | None:
        path = self.cache_dir / key
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        fetched_at = float(payload.get("fetched_at", 0))
        if time.time() - fetched_at > ttl_seconds:
            return None
        return payload

    def _write_cache(self, key: str, payload: Dict[str, object]) -> None:
        path = self.cache_dir / key
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _wikipedia_search_payload(self, query: str, max_results: int) -> Dict[str, object] | None:
        endpoint = (
            "https://en.wikipedia.org/w/api.php?"
            + urlencode(
                {
                    "action": "query",
                    "list": "search",
                    "format": "json",
                    "utf8": "1",
                    "srlimit": str(max_results),
                    "srsearch": query,
                }
            )
        )
        try:
            return json.loads(_http_get(endpoint, self.timeout_seconds))
        except (HTTPError, URLError, TimeoutError, ValueError):
            return None

    def _suggest_anchor_term(self, term: str) -> str:
        cache_key = _cache_key("suggest_v3", term)
        cached = self._read_cache(cache_key, self.search_cache_ttl_seconds)
        if cached:
            return str(cached.get("suggestion", term))

        suggestion = term
        payload = self._wikipedia_search_payload(term, max_results=3)
        if payload is not None:
            searchinfo = payload.get("query", {}).get("searchinfo", {})
            totalhits = int(searchinfo.get("totalhits", 0) or 0)
            search_items = payload.get("query", {}).get("search", [])
            title_contains_term = any(
                term in set(tokenize_retrieval_text(str(item.get("title", "")).strip().lower()))
                for item in search_items[:3]
            )
            raw_suggestion = str(searchinfo.get("suggestion", "")).strip().lower()
            if raw_suggestion and (totalhits == 0 or not title_contains_term):
                candidate = re.sub(r"[^a-z0-9']+", " ", raw_suggestion).strip()
                if candidate and difflib.SequenceMatcher(None, term, candidate).ratio() >= 0.72:
                    suggestion = candidate
            if suggestion == term and (totalhits == 0 or not title_contains_term):
                for item in search_items:
                    title = str(item.get("title", "")).strip().lower()
                    title_terms = [value for value in tokenize_retrieval_text(title) if len(value) > 3]
                    for candidate in title_terms:
                        if difflib.SequenceMatcher(None, term, candidate).ratio() >= 0.78:
                            suggestion = candidate
                            break
                    if suggestion != term:
                        break

        self._write_cache(
            cache_key,
            {
                "fetched_at": time.time(),
                "suggestion": suggestion,
            },
        )
        return suggestion

    def normalize_query(self, query: str) -> str:
        normalized = " ".join(query.strip().split()).lower()
        if not normalized:
            return normalized
        anchor_terms = [
            term
            for term in tokenize_retrieval_text(normalized)
            if term not in FOLLOW_UP_TOPIC_TERMS and len(term) > 4
        ]
        updated = normalized
        for term in anchor_terms:
            suggestion = self._suggest_anchor_term(term)
            if suggestion and suggestion != term:
                updated = re.sub(rf"\b{re.escape(term)}\b", suggestion, updated, count=1)
        return updated

    def search(self, query: str, max_results: int = 4) -> List[WebSearchResult]:
        normalized_query = self.normalize_query(query)
        cache_key = _cache_key("search_v9", normalized_query)
        cached = self._read_cache(cache_key, self.search_cache_ttl_seconds)
        if cached:
            return [WebSearchResult(**item) for item in cached.get("results", [])]

        results = []
        for variant in _search_variants(normalized_query):
            results.extend(self._search_wikipedia(variant, max_results=max_results))
            results.extend(self._search_duckduckgo(variant, max_results=max_results * 2))
            results.extend(self._search_bing(variant, max_results=max_results * 2))
        deduped = self._dedupe_results(normalized_query, results, max_results=max_results)

        self._write_cache(
            cache_key,
            {
                "fetched_at": time.time(),
                "results": [result.__dict__ for result in deduped],
            },
        )
        return deduped

    def _search_duckduckgo(self, query: str, max_results: int) -> List[WebSearchResult]:
        endpoint = "https://duckduckgo.com/html/?" + urlencode({"q": query})
        try:
            html = _http_get(endpoint, self.timeout_seconds)
        except (HTTPError, URLError, TimeoutError):
            return []

        soup = BeautifulSoup(html, "html.parser")
        results: List[WebSearchResult] = []
        cards = soup.select(".result")
        if not cards:
            cards = soup.select(".web-result")
        for index, card in enumerate(cards, start=1):
            link = card.select_one("a.result__a") or card.select_one("a.result-link") or card.select_one("a")
            if link is None:
                continue
            url = _normalize_url(str(link.get("href", "")).strip())
            title = _normalize_text(link.get_text(" ", strip=True))
            if not url or not title or not url.startswith("http"):
                continue
            if any(pattern in url.lower() for pattern in BAD_URL_PATTERNS):
                continue
            snippet_node = card.select_one(".result__snippet") or card.select_one(".result-snippet")
            snippet = _normalize_text(snippet_node.get_text(" ", strip=True) if snippet_node else "")
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    domain=_url_domain(url),
                    rank=index,
                )
            )
            if len(results) >= max_results:
                break
        return results

    def _search_wikipedia(self, query: str, max_results: int) -> List[WebSearchResult]:
        payload = self._wikipedia_search_payload(query, max_results)
        if payload is None:
            return []

        results = []
        for index, item in enumerate(payload.get("query", {}).get("search", []), start=1):
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            snippet_html = str(item.get("snippet", "")).replace("\\/", "/")
            snippet = BeautifulSoup(snippet_html, "html.parser").get_text(" ", strip=True)
            url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=_normalize_text(snippet),
                    domain="wikipedia.org",
                    rank=index,
                )
            )
        return results

    def _search_bing(self, query: str, max_results: int) -> List[WebSearchResult]:
        endpoint = "https://www.bing.com/search?" + urlencode({"q": query, "format": "rss"})
        try:
            xml = _http_get(endpoint, self.timeout_seconds)
        except (HTTPError, URLError, TimeoutError):
            return []

        results = []
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return []

        for index, item in enumerate(root.findall("./channel/item"), start=1):
            title = (item.findtext("title") or "").strip()
            url = _normalize_url((item.findtext("link") or "").strip())
            if not url or not title or not url.startswith("http"):
                continue
            domain = _url_domain(url)
            if any(pattern in url.lower() for pattern in BAD_URL_PATTERNS):
                continue

            snippet = (item.findtext("description") or "").strip()
            results.append(
                WebSearchResult(
                    title=_normalize_text(title),
                    url=url,
                    snippet=_normalize_text(snippet),
                    domain=domain,
                    rank=index,
                )
            )
            if len(results) >= max_results:
                break
        return results

    def _dedupe_results(self, query: str, results: List[WebSearchResult], max_results: int) -> List[WebSearchResult]:
        query_is_medical = is_medical_query(query)
        focus_terms = {
            term
            for term in tokenize_retrieval_text(extract_focus_phrase(query) or query)
            if term not in STOPWORDS and term not in SEARCH_NOISE_TERMS and len(term) > 2
        }
        query_terms = {
            term
            for term in tokenize_retrieval_text(query)
            if term not in STOPWORDS and term not in SEARCH_NOISE_TERMS and len(term) > 2
        }
        topic_terms = {term for term in query_terms if term in FOLLOW_UP_TOPIC_TERMS}
        anchor_terms = query_terms.difference(topic_terms)
        seen_urls = set()
        scored: List[tuple[float, WebSearchResult]] = []
        for result in results:
            normalized_url = result.url.rstrip("/")
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)

            score = 10.0 - float(result.rank)
            score += _domain_weight(result.domain, query_is_medical)
            snippet_terms = set(tokenize_retrieval_text(result.snippet))
            score += float(len(query_terms.intersection(snippet_terms)))
            title_terms = set(tokenize_retrieval_text(result.title))
            score += 0.8 * float(len(query_terms.intersection(title_terms)))
            if focus_terms and focus_terms.issubset(title_terms):
                score += 5.0
            if anchor_terms and anchor_terms.issubset(title_terms):
                score += 5.0
            if anchor_terms:
                anchor_coverage = _anchor_term_coverage(anchor_terms, snippet_terms.union(title_terms))
                if (len(anchor_terms) <= 2 and anchor_coverage < 1.0) or (len(anchor_terms) > 2 and anchor_coverage < 0.6):
                    continue
                score += 1.5 * float(len(anchor_terms.intersection(snippet_terms.union(title_terms))))
            scored.append((score, result))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in scored[:max_results]]

    def fetch_page(self, result: WebSearchResult) -> WebPage | None:
        cache_key = _cache_key("page_v2", result.url)
        cached = self._read_cache(cache_key, self.page_cache_ttl_seconds)
        if cached:
            return WebPage(
                title=str(cached.get("title", result.title)),
                url=str(cached.get("url", result.url)),
                text=str(cached.get("text", "")),
                domain=str(cached.get("domain", result.domain)),
            )

        page = self._fetch_page_uncached(result)
        if page is None:
            return None
        self._write_cache(
            cache_key,
            {
                "fetched_at": time.time(),
                "title": page.title,
                "url": page.url,
                "text": page.text,
                "domain": page.domain,
            },
        )
        return page

    def _fetch_page_uncached(self, result: WebSearchResult) -> WebPage | None:
        if result.domain == "wikipedia.org":
            title = result.url.rsplit("/", 1)[-1]
            summary_url = (
                "https://en.wikipedia.org/w/api.php?"
                + urlencode(
                    {
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": "1",
                        "redirects": "1",
                        "format": "json",
                        "titles": title,
                    }
                )
            )
            try:
                payload = json.loads(_http_get(summary_url, self.timeout_seconds))
            except (HTTPError, URLError, TimeoutError, ValueError):
                return None
            pages = payload.get("query", {}).get("pages", {})
            page_payload = next(iter(pages.values()), {}) if isinstance(pages, dict) else {}
            extract = _normalize_text(str(page_payload.get("extract", "")))
            if not extract:
                return None
            return WebPage(
                title=str(page_payload.get("title", result.title)),
                url=result.url,
                text=extract,
                domain=result.domain,
            )

        try:
            html = _http_get(result.url, self.timeout_seconds)
        except (HTTPError, URLError, TimeoutError):
            return None

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
            tag.decompose()

        root = soup.find("article") or soup.find("main") or soup.body or soup
        text_parts = []
        seen = set()
        for node in root.find_all(["h1", "h2", "h3", "p", "li"]):
            text = _normalize_text(node.get_text(" ", strip=True))
            if len(text) < 40 or text in seen:
                continue
            seen.add(text)
            text_parts.append(text)
            if sum(len(part) for part in text_parts) > 8000:
                break

        page_text = "\n".join(text_parts).strip()
        if not page_text and result.snippet:
            page_text = result.snippet
        if not page_text:
            return None

        title = _normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else result.title
        return WebPage(
            title=title or result.title,
            url=result.url,
            text=page_text,
            domain=result.domain,
        )

    def retrieve(self, query: str, max_results: int = 4) -> List[Dict[str, str | float]]:
        results = self.search(query, max_results=max_results)
        if not results:
            return []

        with ThreadPoolExecutor(max_workers=min(4, max_results)) as executor:
            pages = list(executor.map(self.fetch_page, results))

        focus_terms = {
            term
            for term in tokenize_retrieval_text(extract_focus_phrase(query) or query)
            if term not in STOPWORDS and term not in SEARCH_NOISE_TERMS and len(term) > 2
        }
        query_terms = {
            term
            for term in tokenize_retrieval_text(query)
            if term not in STOPWORDS and term not in SEARCH_NOISE_TERMS and len(term) > 2
        }
        topic_terms = {term for term in query_terms if term in FOLLOW_UP_TOPIC_TERMS}
        anchor_terms = query_terms.difference(topic_terms)
        scored_chunks: List[tuple[float, Dict[str, str | float]]] = []
        for result, page in zip(results, pages):
            if page is None:
                continue
            for chunk in chunk_text_by_words(page.text, chunk_words=220, overlap_words=40):
                chunk_terms = set(tokenize_retrieval_text(chunk))
                if not chunk_terms:
                    continue
                overlap_terms = query_terms.intersection(chunk_terms)
                overlap = len(overlap_terms)
                if overlap == 0 and result.snippet:
                    snippet_terms = set(tokenize_retrieval_text(result.snippet))
                    overlap = len(query_terms.intersection(snippet_terms))
                    overlap_terms = query_terms.intersection(snippet_terms)
                if overlap == 0:
                    continue
                if anchor_terms:
                    anchor_coverage = _anchor_term_coverage(anchor_terms, chunk_terms.union(set(tokenize_retrieval_text(result.title))))
                    if (len(anchor_terms) <= 2 and anchor_coverage < 1.0) or (len(anchor_terms) > 2 and anchor_coverage < 0.6):
                        continue
                score = float(overlap)
                if topic_terms and topic_terms.intersection(overlap_terms):
                    score += 2.0
                title_terms = set(tokenize_retrieval_text(result.title))
                if focus_terms and focus_terms.issubset(title_terms):
                    score += 4.0
                if anchor_terms and anchor_terms.issubset(title_terms):
                    score += 4.0
                trust_tier = _domain_weight(result.domain, is_medical_query(query))
                score += min(4.0, trust_tier / 2.5)
                scored_chunks.append(
                    (
                        score,
                        {
                            "text": chunk,
                            "source": page.url,
                            "url": page.url,
                            "domain": "web",
                            "score": round(score, 4),
                            "title": page.title,
                            "web_domain": result.domain,
                            "trust_tier": round(trust_tier, 2),
                            "source_type": "web",
                        },
                    )
                )

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in scored_chunks[:max_results]]
