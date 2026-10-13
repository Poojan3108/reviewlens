"""LLM step (Groq): choose product aspects, summarize each expert article's verdict,
and tag which aspects every review praises or criticizes, with exact supporting quotes."""
import json
import os
import re
import time
from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

MODEL_NAME = "openai/gpt-oss-120b"


def _chat() -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=0, max_tokens=4000, reasoning_effort="low",
                    api_key=os.environ["GROQ_API_KEY"])


def _parse_json(schema, text: str):
    """Pull the first JSON object out of a plain-text reply and validate it."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return schema.model_validate_json(text[start:end + 1])


def _call(schema, prompt: str, retries: int = 3):
    """Ask for structured output, trying the most reliable method first:
    1. Groq's strict JSON-schema mode
    2. Plain JSON instructions, parsed and validated ourselves
    3. Tool calling
    Retries the whole chain on rate limits."""
    attempts = [
        lambda: _chat().with_structured_output(schema, method="json_schema").invoke(prompt),
        lambda: _parse_json(schema, _chat().invoke(
            prompt + "\n\nRespond with ONLY a JSON object (no other text) matching this JSON schema:\n"
            + json.dumps(schema.model_json_schema())).content),
        lambda: _chat().with_structured_output(schema, method="function_calling").invoke(prompt),
    ]
    last_error = None
    for round_ in range(retries):
        for attempt in attempts:
            try:
                result = attempt()
                if result is not None:
                    return result
            except Exception as e:  # noqa: BLE001
                last_error = e
                if "rate" in str(e).lower() or "429" in str(e):
                    time.sleep(10)
                    break  # rate limited: wait, then restart from the best method
        else:
            time.sleep(2)
    raise last_error


# ---------- Schemas ----------
class Aspects(BaseModel):
    aspects: list[str] = Field(description="4-6 short aspect names, e.g. 'Sound quality', 'Battery life'")


class Mention(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str = Field(description="Exact words copied from the text, under 25 words")


class ExpertAnalysis(BaseModel):
    reviews_this_product: bool = Field(description="False if the article is not a review of this exact product")
    verdict: str = Field(description="3-5 sentences in the reviewer's own words summarizing their opinion, including praise and complaints")
    stated_score: float | None = Field(default=None, description="The reviewer's own overall score converted to a 5-point scale, or null if none given")
    mentions: list[Mention]


class ReviewMentions(BaseModel):
    review_number: int
    mentions: list[Mention]


class CustomerBatch(BaseModel):
    reviews: list[ReviewMentions]


# ---------- Helpers ----------
def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def grounded(mentions: list[Mention], source_text: str, aspects: list[str]) -> list[dict]:
    """Keep only mentions whose quote really appears in the source and whose aspect is on our list."""
    text = _norm(source_text)
    allowed = {a.lower(): a for a in aspects}
    keep = []
    for m in mentions:
        aspect = allowed.get(m.aspect.strip().lower())
        if aspect and _norm(m.quote) and _norm(m.quote) in text:
            keep.append({"aspect": aspect, "sentiment": m.sentiment, "quote": m.quote.strip()})
    return keep


# ---------- Steps ----------
def choose_aspects(product: str, sample_texts: list[str]) -> list[str]:
    sample = "\n---\n".join(t[:600] for t in sample_texts[:8])
    result = _call(Aspects, (
        f"Product: {product}\n\nHere are excerpts from reviews of it:\n{sample}\n\n"
        "List the 4-6 aspects shoppers care about most for THIS kind of product "
        "(e.g. for headphones: Sound quality, Comfort, Battery life, Noise cancelling, Price/value). "
        "Use short title-case names. Always include 'Price/value'."))
    aspects = [a.strip() for a in result.aspects if a.strip()][:6]
    return aspects or ["Quality", "Price/value"]


def analyze_expert(product: str, aspects: list[str], article: dict) -> dict | None:
    result = _call(ExpertAnalysis, (
        f"Product: {product}\nAspects to look for: {', '.join(aspects)}\n\n"
        f"Article title: {article['title']}\nArticle text:\n{article['text']}\n\n"
        "1) Is this article a review of this exact product? (Comparison/roundup articles count "
        "only if they evaluate this product specifically.)\n"
        "2) Summarize the reviewer's opinion of the product in 3-5 sentences, staying close to "
        "their own wording.\n"
        "3) If the reviewer gives an overall score (e.g. 4/5, 8.5/10, 90%), convert it to a 5-point scale.\n"
        "4) For each aspect the article discusses, give the sentiment and an EXACT quote "
        "copied word-for-word from the article. Only use the aspect names listed above."))
    if not result.reviews_this_product:
        return None
    score = result.stated_score if result.stated_score and 0 < result.stated_score <= 5 else None
    return {"verdict": result.verdict, "stated_score": score,
            "mentions": grounded(result.mentions, article["text"], aspects)}


def analyze_customers(product: str, aspects: list[str], reviews: list[str],
                      batch_size: int = 15) -> list[list[dict]]:
    """Returns one list of aspect mentions per review (same order as input)."""
    out = [[] for _ in reviews]
    for start in range(0, len(reviews), batch_size):
        batch = reviews[start:start + batch_size]
        numbered = "\n".join(f"[{i + 1}] {r[:800]}" for i, r in enumerate(batch))
        result = _call(CustomerBatch, (
            f"Product: {product}\nAspects: {', '.join(aspects)}\n\nCustomer reviews:\n{numbered}\n\n"
            "For each review number, list which of the aspects it mentions, the sentiment, and an "
            "EXACT quote copied word-for-word from that review. Skip aspects a review doesn't "
            "mention. Only use the aspect names listed above."))
        for item in result.reviews:
            idx = item.review_number - 1
            if 0 <= idx < len(batch):
                out[start + idx] = grounded(item.mentions, batch[idx], aspects)
    return out