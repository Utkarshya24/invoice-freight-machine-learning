import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

def load_vendor_invoice_data(db_path: str):
    """
    load vendor invoice data from SQlite database.
    """
    conn = sqlite3.connect(db_path)
    query = "select * from vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def prepare_features(df: pd.DataFrame):
    """
    select features and target variable.
    """
    x = df[["Dollars"]]
    y = df[["Freight"]]
    return x, y 

def split_data(x,y, test_size=0.2, random_state=42):
    """
    split dataset into train and test sets.
    """
    return train_test_split(
        x,y, test_size=test_size, random_state=random_state
    )