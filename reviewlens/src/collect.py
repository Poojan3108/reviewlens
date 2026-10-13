"""Gather expert reviews and US store prices from the web, and clean pasted customer reviews."""
import os
import re
from urllib.parse import urlparse

from tavily import TavilyClient

STORES = {
    "amazon.com": "Amazon", "bestbuy.com": "Best Buy", "walmart.com": "Walmart",
    "target.com": "Target", "bhphotovideo.com": "B&H Photo", "newegg.com": "Newegg",
    "costco.com": "Costco", "ebay.com": "eBay",
}
# Not expert reviews: stores, social media, and video sites
NOT_EXPERT = list(STORES) + [
    "youtube.com", "tiktok.com", "instagram.com", "facebook.com", "x.com", "twitter.com",
    "pinterest.com", "quora.com", "linkedin.com",
]
MAX_ARTICLE_CHARS = 6000


def _client() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def search_expert_reviews(product: str, max_results: int = 8) -> list[dict]:
    res = _client().search(
        f"{product} review", search_depth="advanced", max_results=max_results,
        include_raw_content=True, exclude_domains=NOT_EXPERT)
    articles = []
    for r in res.get("results", []):
        text = (r.get("raw_content") or r.get("content") or "").strip()
        if len(text) < 400:
            continue
        articles.append({"title": r.get("title", ""), "url": r["url"], "source": domain(r["url"]),
                         "text": text[:MAX_ARTICLE_CHARS]})
    return articles


PRICE = re.compile(r"\$\s?(\d{1,2},\d{3}|\d{1,5})(\.\d{2})?")


def _key_tokens(product: str) -> tuple[list[str], list[str]]:
    """Model tokens (contain a digit, e.g. 'wh-1000xm5') and other words, lowercased."""
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9\-]+", product.lower()) if len(w) >= 3]
    return [w for w in words if any(c.isdigit() for c in w)], [w for w in words if not any(c.isdigit() for c in w)]


def is_product_listing(product: str, title: str, url: str) -> bool:
    """Reject store pages that aren't about this product (careers, investor pages, ads...)."""
    hay = re.sub(r"[^a-z0-9]", "", f"{title} {url}".lower())
    models, words = _key_tokens(product)
    if models:
        return any(re.sub(r"[^a-z0-9]", "", m) in hay for m in models)
    return sum(w in hay for w in words) >= max(1, len(words) // 2)


def search_offers(product: str) -> list[dict]:
    res = _client().search(f"{product} price", max_results=10, include_domains=list(STORES))
    offers, seen = [], set()
    for r in res.get("results", []):
        d = domain(r["url"])
        store = next((name for key, name in STORES.items() if d.endswith(key)), None)
        if not store or r["url"] in seen:
            continue
        seen.add(r["url"])
        price = None
        for m in PRICE.finditer(f"{r.get('title', '')} {r.get('content', '')}"):
            value = float(m.group(1).replace(",", "") + (m.group(2) or ""))
            if 5 <= value <= 20000:
                price = value
                break
        title = (r.get("title") or "").strip()
        if not is_product_listing(product, title, r["url"]):
            continue
        if len(title) < 12 or re.search(r"prime modal|sign in|captcha|robot check", title, re.I):
            title = f"{product} on {store}"
        offers.append({"store": store, "title": title, "url": r["url"], "price": price})
    # keep at most 2 listings per store
    kept, per_store = [], {}
    for o in offers:
        per_store[o["store"]] = per_store.get(o["store"], 0) + 1
        if per_store[o["store"]] <= 2:
            kept.append(o)
    return kept


# Lines that shoppers copy along with Amazon / Best Buy / Walmart reviews
NOISE = re.compile(
    r"^(verified purchase|helpful|report|read more|see more|translate review.*|"
    r"\d+ (people|person) found this helpful|one person found this helpful|"
    r"reviewed in .* on .*|"
    r"(size|color|style|colour|pattern|configuration): .*|was this review helpful.*|"
    r"yes ?\(\d+\)|no ?\(\d+\)|posted .* ago|\d+ (days|months|years) ago)$",
    re.I)


STAR_LINE = re.compile(r"^(?:rated\s+)?(\d(?:\.\d)?)\s+out of\s+5(?:\s+stars?)?\s*(.*)$", re.I)


SEPARATOR = re.compile(r"^\s*(-{3,}|={3,}|\*{3,})\s*$", re.M)


def _clean_block(block: str):
    stated, lines = None, []
    for ln in (x.strip() for x in block.split("\n")):
        if not ln or NOISE.match(ln):
            continue
        star = STAR_LINE.match(ln)
        if star:
            stated = float(star.group(1))
            if star.group(2):
                lines.append(star.group(2).strip() + ".")
            continue
        lines.append(ln)
    return " ".join(lines).strip(), stated


def split_pasted_reviews(text: str) -> list[dict]:
    """Split pasted text into separate reviews.

    1. If reviews include star lines ("4.0 out of 5 stars ..."), each star line starts a new
       review, so multi-paragraph reviews stay whole.
    2. Otherwise, a line of '---' separates reviews.
    3. Otherwise, blank lines separate reviews.
    Clutter like 'Verified Purchase' is removed and pasted star ratings are kept."""
    text = text.replace("\r", "")
    lines = text.split("\n")
    star_idx = [i for i, ln in enumerate(lines) if STAR_LINE.match(ln.strip())]
    if len(star_idx) >= 2:
        bounds = star_idx + [len(lines)]
        blocks = ["\n".join(lines[bounds[i]:bounds[i + 1]]) for i in range(len(star_idx))]
        if star_idx[0] > 0:  # text before the first star line
            blocks.insert(0, "\n".join(lines[:star_idx[0]]))
    elif SEPARATOR.search(text):
        blocks = SEPARATOR.split(text)
    else:
        blocks = re.split(r"\n\s*\n", text)
    reviews = []
    for block in blocks:
        if not block or SEPARATOR.fullmatch(block.strip() or "x"):
            continue
        review, stated = _clean_block(block)
        if len(review) >= 15:
            reviews.append({"text": review, "stated_stars": stated})
    return reviews