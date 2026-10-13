-- name: rating_by_source_type
-- Expert vs customer rating, kept separate on purpose.
SELECT source_type,
       ROUND(AVG(stars), 2)  AS avg_stars,
       COUNT(*)              AS reviews,
       ROUND(AVG(stated_score), 2) AS avg_stated_score
FROM reviews
WHERE search_id = :search_id
GROUP BY source_type;

-- name: rating_by_source
SELECT source_type, source,
       ROUND(AVG(stars), 2) AS avg_stars,
       COUNT(*)             AS reviews
FROM reviews
WHERE search_id = :search_id
GROUP BY source_type, source
ORDER BY source_type, reviews DESC;

-- name: aspect_scores
-- Net sentiment per aspect mapped to a 1-5 scale: 3 + 2 * (positive - negative) / mentions.
SELECT r.source_type,
       a.aspect,
       COUNT(*)                                              AS mentions,
       SUM(a.sentiment = 'positive')                         AS positive,
       SUM(a.sentiment = 'negative')                         AS negative,
       ROUND(3 + 2.0 * (SUM(a.sentiment = 'positive') - SUM(a.sentiment = 'negative'))
                 / COUNT(*), 2)                              AS score
FROM aspect_mentions a
JOIN reviews r ON r.id = a.review_id
WHERE r.search_id = :search_id
GROUP BY r.source_type, a.aspect
ORDER BY r.source_type, mentions DESC;

-- name: top_quotes
-- Up to 3 example quotes per aspect and sentiment, preferring readable lengths (40-220 chars).
SELECT aspect, sentiment, quote, source, url, source_type
FROM (
    SELECT a.aspect, a.sentiment, a.quote, r.source, r.url, r.source_type,
           ROW_NUMBER() OVER (PARTITION BY a.aspect, a.sentiment
                              ORDER BY (LENGTH(a.quote) BETWEEN 40 AND 220) DESC,
                                       LENGTH(a.quote) DESC) AS rn
    FROM aspect_mentions a
    JOIN reviews r ON r.id = a.review_id
    WHERE r.search_id = :search_id AND a.quote IS NOT NULL AND a.sentiment != 'neutral'
)
WHERE rn <= 3
ORDER BY aspect, sentiment, rn;

-- name: star_distribution
SELECT source_type, CAST(ROUND(stars) AS INTEGER) AS star, COUNT(*) AS reviews
FROM reviews
WHERE search_id = :search_id
GROUP BY source_type, star
ORDER BY source_type, star;

-- name: offers
SELECT store, title, url, price
FROM offers
WHERE search_id = :search_id
ORDER BY price IS NULL, price;

-- name: recent_search
-- Cache: reuse an expert search for the same product from the last 24 hours.
SELECT id FROM searches
WHERE LOWER(product) = LOWER(:product)
  AND created_at >= datetime('now', '-1 day')
  AND EXISTS (SELECT 1 FROM reviews WHERE search_id = searches.id AND source_type = 'expert')
ORDER BY id DESC LIMIT 1;