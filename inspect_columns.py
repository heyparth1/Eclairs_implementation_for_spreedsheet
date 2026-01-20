import pandas as pd
import glob
import os

# Find the excel file
files = glob.glob("*.xls*")
if not files:
    print("No Excel file found")
else:
    df = pd.read_excel(files[0])
    print("Columns:", df.columns.tolist())
    print("\n--- SAMPLE Product Description ---")
    if 'Product Description' in df.columns:
        print(df['Product Description'].head(5))
    
    print("\n--- SAMPLE Type/SNo ---")
    if 'Type/SNo' in df.columns:
        print(df['Type/SNo'].head(5))
        
    print("\n--- SAMPLE Product ID (if exists) ---")
    if 'Product ID' in df.columns:
        print(df['Product ID'].head(5))
