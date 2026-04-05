import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

def load_invoice_data():
    conn = sqlite3.connect('data/inventory.db')

    query = """
    WITH purchase_agg AS ( 
      select 
          p.PONumber,
          count(distinct p.Brand) as total_brands,
          sum(p.Quantity) as total_item_quantity,
          sum(p.Dollars) as total_item_dollars,  -- already present
          avg(julianday(p.ReceivingDate) - julianday(p.PODate)) as avg_receiving_delay
      from purchases p
      group by p.PONumber
    )

    select 
        vi.Quantity as invoice_quantity,
        vi.Dollars as invoice_dollars,
        vi.Freight,
        (julianday(vi.Invoicedate) - julianday(vi.PODate)) as days_po_to_invoice,
        (julianday(vi.payDate) - julianday(vi.Invoicedate)) as days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,  
        pa.avg_receiving_delay

    from vendor_invoice vi
    LEFT JOIN purchase_agg pa 
        ON vi.PONumber = pa.PONumber
  """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    if (abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5 ):
        return 1 
        
    if row["avg_receiving_delay"] > 10:
        return 1 
    return 0

def apply_labels(df):
  df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
  return df

def split_data(df, features, target):
    x = df[features]
    y = df[target]

    return train_test_split(
        x, y , test_size=0.2, random_state=42
    )

def scale_features(x_train, x_test, scaler_path):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(scaler, "models/scaler.pkl")
    return x_train_scaled, x_test_scaled