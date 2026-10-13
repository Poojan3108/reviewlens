# ⭐ ReviewLens — Expert & Customer Reviews, Side by Side

Type any product and ReviewLens finds expert reviews across the web, scores them with a sentiment model trained on **42,411 real product reviews**, breaks opinions down by aspect (battery, comfort, sound, price...), and shows pros and cons with quotes linked to their sources. Paste reviews from Amazon, Best Buy, Walmart, and more to add the customer view, and see where to buy it in the US.

**Live demo:** [add your Streamlit link here]

## Why

Shoppers face hundreds of reviews spread across dozens of sites, and experts and customers often disagree: experts test a product for days, while customers live with it for months. ReviewLens puts both views side by side, never blended into one number, so you can see where they agree and where they don't.

## Features

- **Expert rating and customer rating, shown separately**, with a per-source breakdown (e.g. rtings.com vs. Amazon vs. Best Buy)
- **Aspect analysis** discovered automatically for any product type, scored separately for experts and customers
- **Pros and cons with verified quotes:** every quote is checked to appear word-for-word in its source, with links
- **Where to buy:** US store listings (Amazon, Best Buy, Walmart, Target, B&H, Newegg, Costco, eBay) with prices seen in search results
- **Paste or upload customer reviews** from any store; star ratings and clutter like "Verified Purchase" are handled automatically
- **Built-in model check:** when pasted reviews include the customer's own star rating, the app reports how close the model's predictions were

## How it works

```
Product name ─► Tavily web search ─► expert articles ─► LLM summarizes each verdict ─┐
Pasted reviews ─► cleaning & splitting ─────────────────────────────────────────────┤
                                                                                     ▼
                        Sentiment model (TF-IDF + Logistic Regression) ─► 1-5 star rating
                        LLM aspect tagging ─► quote verification ─► SQLite ─► SQL analytics ─► Streamlit
```

1. **Collection** (`src/collect.py`): Tavily searches for expert reviews (excluding stores and social media) and for US store listings; pasted reviews are split and cleaned.
2. **Sentiment model** (`training/train_sentiment.py`, `src/scoring.py`): predicts the probability of each star rating and reports the expected value.
3. **LLM step** (`src/extract.py`, Groq): chooses the 4-6 aspects that matter for this product, summarizes each expert article's verdict, tags aspect sentiment, and extracts quotes. Off-topic articles are filtered out, and any quote not found word-for-word in its source is discarded.
4. **SQL** (`sql/schema.sql`, `sql/queries.sql`): reviews, aspect mentions, and prices are stored in SQLite; all ratings, aspect scores, and best quotes are computed with SQL (aggregations, joins, `ROW_NUMBER()` window functions). Searches are cached for 24 hours.

## The sentiment model

Trained on 42,411 reviews from three very different product categories (food, music & audio gear, clothing) using public Kaggle datasets.

| Test | Exact star | Within 1 star | Positive / neutral / negative correct | Avg error (stars) |
|---|---|---|---|---|
| Random 20% hold-out | 66.5% | 93.5% | 84.5% | 0.62 |
| Unseen category: food | 65.8% | 89.2% | 82.0% | 0.83 |
| Unseen category: music & audio | 61.3% | 90.4% | 84.2% | 0.86 |
| Unseen category: clothing | 59.1% | 87.9% | 76.8% | 0.75 |

The **unseen category** tests train on two categories and test on the third, to check how well the model handles product types it has never seen, which matters because users can search for any product.

## Where reviews come from

- **Found automatically:** expert review sites and tech blogs (e.g. RTINGS, TechRadar, CNET, Tom's Guide, PCMag), plus some forums.
- **Paste or upload:** Amazon, Best Buy, Walmart, Target, eBay, Flipkart, and other stores, which don't allow automated collection of their reviews.

## Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY and TAVILY_API_KEY
streamlit run app.py
```

To retrain the sentiment model: `python training/train_sentiment.py` (downloads the training data automatically).

## Tech stack

Python · scikit-learn · Groq (GPT-OSS-120B) · LangChain · Pydantic structured outputs · Tavily Search API · SQLite · Streamlit · Altair

## Limitations

- Prices come from search results and may be outdated or refer to a different variant.
- The model was trained on customer reviews; expert articles are summarized by the LLM first so the model scores the reviewer's verdict rather than the whole article.
- Automatic search finds mostly expert reviews; customer reviews from major stores must be pasted.
