import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("news_dataset.csv")

print("Columns:", df.columns)
print("Total rows:", len(df))

# Keep only required columns
df = df[['text', 'label']]

# Remove null values
df.dropna(inplace=True)

# Ensure labels are integers
df['label'] = df['label'].astype(int)

print("Label count before balancing:\n", df['label'].value_counts())

# 🔥 Balance dataset (UPSAMPLING)
df_fake = df[df.label == 0]
df_real = df[df.label == 1]

df_real_upsampled = resample(
    df_real,
    replace=True,
    n_samples=len(df_fake),
    random_state=42
)

df = pd.concat([df_fake, df_real_upsampled])

# Shuffle dataset
df = df.sample(frac=1, random_state=42)

print("Label count after balancing:\n", df['label'].value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# 🔥 Improved Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.7,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔥 Improved Model
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Train model
model.fit(X_train_vec, y_train)

# Accuracy check
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model & vectorizer
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("Model trained successfully!")