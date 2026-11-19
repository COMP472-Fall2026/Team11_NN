from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

image_train = pd.read_csv(DATA_PATH / "image_train.csv")
image_test = pd.read_csv(DATA_PATH / "image_test.csv")

sentiment_train = pd.read_csv(DATA_PATH / "sentiment_train.csv")
sentiment_test = pd.read_csv(DATA_PATH / "sentiment_test.csv")

sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

def split_dataset(df):
    feature_columns = [
        column for column in df.columns if column.startswith("feature_")
    ]

    X = df[feature_columns].to_numpy(dtype="float32")
    y = df["label"].to_numpy(dtype="int64")

    return train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

X_img_train, X_img_val, y_img_train, y_img_val = split_dataset(image_train)
X_sent_train, X_sent_val, y_sent_train, y_sent_val = split_dataset(sentiment_train)