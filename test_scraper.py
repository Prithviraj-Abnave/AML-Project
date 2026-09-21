from ml.scraper import scrape_reviews
import logging

logging.basicConfig(level=logging.INFO)

urls = [
    "https://www.flipkart.com/apple-iphone-15-black-128-gb/p/itm6ac6485515ae4?pid=MOBGTAGPTB3VS24W",
    "https://www.snapdeal.com/product/apple-iphone-14-128gb/674920888066"
]

try:
    df = scrape_reviews(urls, max_pages=1)
    print(df.head())
    print("Total reviews scraped:", len(df))
except Exception as e:
    import traceback
    traceback.print_exc()
