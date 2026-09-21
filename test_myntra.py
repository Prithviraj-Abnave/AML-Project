from ml.scraper import scrape_myntra
import logging

logging.basicConfig(level=logging.INFO)
url = "https://www.myntra.com/reviews/12027436?p=2"
try:
    with open("ml/scraper.py", "r") as f:
        content = f.read()
    with open("ml/scraper.py", "w") as f:
        f.write(content.replace('wait_until="networkidle"', 'wait_until="domcontentloaded"'))
    reviews = scrape_myntra(url, max_scrolls=2)
    print(f"Scraped {len(reviews)} reviews from Myntra.")
except Exception as e:
    pass
