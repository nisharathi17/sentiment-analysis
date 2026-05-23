import pandas as pd
# Load the CSV file into a DataFrame
df = pd.read_csv("IMDB dataset.csv")
# That's it! df now holds all 50,000 rows
print("Loaded!", len(df), "rows")
# See the first 5 rows
df.head()
# See the last 5 rows
df.tail()
# How many rows and columns?
print(df.shape)   # (50000, 2)
# Column names and data types
print(df.info())# Convert labels to numbers

#create a new column called "label" where positive=1 and negative=0
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Save the DataFrame as a new CSV file. index=False means don't write the row numbers (0, 1, 2...) into the file — we don't need them.
df.to_csv("cleaned_reviews.csv", index=False)
print("Saved cleaned_reviews.csv!")

# Verify it worked.Count how many 1s and how many 0s are in the label column. Should print 25000 each — confirming the conversion worked.
print(df["label"].value_counts())
# Access a single column
df["sentiment"]

# Count how many positive vs negative
print(df["sentiment"].value_counts())

# For every review, count how many characters it has. Then show stats — shortest review, longest review, average length. .str.len() = length of text. .describe() = show me the stats summary.
df["review"].str.len().describe()
# Get only positive reviews
positives = df[df["sentiment"] == "positive"]
# Get only negative reviews
negatives = df[df["sentiment"] == "negative"]
# See a random review

