import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_rows=100, output_path='data/synthetic_reviews.csv'):
    # Define columns
    columns = [
        'id', 'dateAdded', 'dateUpdated', 'name', 'asins', 'brand', 'categories',
        'primaryCategories', 'imageURLs', 'keys', 'manufacturer', 'manufacturerNumber',
        'reviews.date', 'reviews.dateAdded', 'reviews.dateSeen', 'reviews.doRecommend',
        'reviews.id', 'reviews.numHelpful', 'reviews.rating', 'reviews.sourceURLs',
        'reviews.text', 'reviews.title', 'reviews.username', 'sourceURLs'
    ]

    # Sample data generators
    names = ['SuperPhone 2000', 'Quantum Tablet X', 'NoiseCanceling Headphones Z', 'SmartWatch Elite']
    brands = ['TechCorp', 'Innova', 'AcousticsPro']
    categories = ['Electronics', 'Computers', 'Audio']
    
    positive_reviews = [
        "This product is amazing. The battery life is great and the screen is beautiful.",
        "I love this! Highly recommended to everyone. Best purchase of the year.",
        "Works perfectly out of the box. Setup was a breeze and the performance is stellar.",
        "Great value for the price. I've been using it every day without any issues.",
        "The sound quality is incredible. Bass is punchy and highs are crisp."
    ]
    
    negative_reviews = [
        "Terrible experience. The device stopped working after a week.",
        "Battery drains too fast. Not what I expected for this price.",
        "Customer service was unhelpful and the product feels cheap.",
        "Screen cracked easily. Very fragile build quality.",
        "Software is buggy and crashes often. Needs a major update."
    ]
    
    mixed_reviews = [
        "It's decent, but there are better options out there. Average performance.",
        "Good hardware, but the software needs some work. 3 stars.",
        "Gets the job done, though it's a bit overpriced for what it offers."
    ]
    
    titles = ["Great!", "Awful", "Could be better", "Amazing product", "Do not buy", "Average"]
    usernames = ['user123', 'techgeek', 'gadgetlover', 'disappointed1', 'happy_shopper']

    data = []
    
    for i in range(num_rows):
        product_idx = random.randint(0, len(names)-1)
        rating = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.1, 0.2, 0.3, 0.3])[0]
        
        if rating >= 4:
            text = random.choice(positive_reviews)
            do_rec = "TRUE"
        elif rating <= 2:
            text = random.choice(negative_reviews)
            do_rec = "FALSE"
        else:
            text = random.choice(mixed_reviews)
            do_rec = random.choice(["TRUE", "FALSE"])

        row = {
            'id': f'SYN_{i}',
            'dateAdded': (datetime.now() - timedelta(days=random.randint(100, 300))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'dateUpdated': datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            'name': names[product_idx],
            'asins': f'B000SYN{product_idx}',
            'brand': random.choice(brands),
            'categories': categories[product_idx % len(categories)],
            'primaryCategories': 'Electronics',
            'imageURLs': 'http://example.com/image.jpg',
            'keys': f'syn_key_{i}',
            'manufacturer': random.choice(brands),
            'manufacturerNumber': f'MN_{i}',
            'reviews.date': (datetime.now() - timedelta(days=random.randint(1, 100))).strftime("%Y-%m-%dT00:00:00.000Z"),
            'reviews.dateAdded': '',
            'reviews.dateSeen': '',
            'reviews.doRecommend': do_rec,
            'reviews.id': '',
            'reviews.numHelpful': random.randint(0, 50),
            'reviews.rating': rating,
            'reviews.sourceURLs': 'http://example.com/review',
            'reviews.text': text,
            'reviews.title': random.choice(titles),
            'reviews.username': random.choice(usernames),
            'sourceURLs': 'http://example.com/product'
        }
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_rows} rows of synthetic data at {output_path}")

if __name__ == '__main__':
    generate_synthetic_data(20000, 'c:/Users/admin/Desktop/SEM 7/Advance ML/Project/data/synthetic_reviews.csv')
