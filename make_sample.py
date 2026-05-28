import pandas as pd

data = {
    'product_id': ['B001','B001','B002','B002','B003','B003','B004','B004'],
    'review_text': [
        "Great product, works perfectly, very happy!",
        "Broke after one week. Terrible quality. Waste of money.",
        "Decent for the price, nothing special but does the job.",
        "Absolutely love it, best purchase I have ever made!",
        "Arrived damaged. Very disappointed with the packaging.",
        "Amazing quality, fast delivery, exceeded my expectations!",
        "Completely useless. Does not work as described.",
        "Fantastic product, would highly recommend to everyone!"
    ],
    'star_rating': [5, 1, 3, 5, 1, 5, 1, 5]
}

df = pd.DataFrame(data)
df.to_csv('sample_reviews.csv', index=False)
print("sample_reviews.csv created!")
print(df)