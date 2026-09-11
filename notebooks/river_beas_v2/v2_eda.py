import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("Starting Data Processing Pipeline...")

# 1. File Paths
file_paths = [
    r"C:\Users\omen\OneDrive\Documents\wq_data_hp\Kunah Khad Ds Hamirpur.xlsx", 
    r"C:\Users\omen\OneDrive\Documents\wq_data_hp\River_Beas_Ds_Mandi_Town.xlsx",
    r"C:\Users\omen\OneDrive\Documents\wq_data_hp\RiverBeas Ds Kullu..xlsx",
    r"C:\Users\omen\OneDrive\Documents\wq_data_hp\RiverBeas Ds Nadaun..xlsx",
    r"C:\Users\omen\OneDrive\Documents\wq_data_hp\RiverBeas Ds Pong Dam..xlsx"
]

# 2. Load and Combine Data Safely
df_list = []
for file in file_paths:
    if os.path.exists(file):
        temp_df = pd.read_excel(file) 
        temp_df.columns = temp_df.iloc[0] # Set first row as header
        temp_df = temp_df.drop(0).reset_index(drop=True)
        # Clean column names (strip extra spaces)
        temp_df.columns = [str(col).strip() for col in temp_df.columns]
        df_list.append(temp_df)
    else:
        print(f"Warning: {file} not found.")

df = pd.concat(df_list, ignore_index=True)
print(f"Merged dataset has {df.shape[0]} rows and {df.shape[1]} columns")

# 3. Convert Data to Numeric
# Added TEMP.(ºC) and CL(mg/l) to be converted to numbers
numeric_cols = ['pH', 'TURBIDITY(NTU)', 'COND.(µS/cm)', 'DO(mg/l)', 'BOD(mg/l)', 'COD(mg/l)', 'TEMP.(ºC)', 'CL(mg/l)']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Calculate missing TDS
df['TDS'] = df['COND.(µS/cm)'] * 0.65

# 5. Filter Core Columns (Including Temp and Chlorides now)
final_columns = [
    'pH',
    'TURBIDITY(NTU)',
    'COND.(µS/cm)',
    'TDS',            
    'DO(mg/l)',
    'TEMP.(ºC)',    # Kept: Hardware has Temp
    'CL(mg/l)',     # Kept: Hardware proxy for Chlorides
    'BOD(mg/l)',    # Kept: Hardware proxy for BOD
    'COD(mg/l)'     # Kept temporarily just to calculate the score
]

# Only keep columns that actually exist in the dataframe
final_columns = [c for c in final_columns if c in df.columns]
df = df[final_columns]
df = df.dropna(how='all')

# 6. Handle Missing Values
print("\n--- Cleaning Data ---")
df = df.dropna(subset=['BOD(mg/l)', 'COD(mg/l)'])
print(f"Dropped rows missing BOD/COD. Remaining rows: {len(df)}")

# Fill everything else with the median
columns_to_fill = ['pH', 'TURBIDITY(NTU)', 'COND.(µS/cm)', 'TDS', 'DO(mg/l)', 'TEMP.(ºC)', 'CL(mg/l)']
columns_to_fill = [c for c in columns_to_fill if c in df.columns]
df[columns_to_fill] = df[columns_to_fill].fillna(df[columns_to_fill].median())

print("Null % age after cleaning:")
print((df.isnull().sum() / len(df)) * 100)

# 7. Calculate Quality Labels
def calculate_river_quality(row):
    scores = []
    # BOD
    bod = row['BOD(mg/l)']
    if bod <= 2: scores.append(100)
    elif bod <= 3: scores.append(80)
    elif bod <= 5: scores.append(60)
    elif bod <= 10: scores.append(30)
    else: scores.append(10)
        
    # COD
    cod = row['COD(mg/l)']
    if cod <= 10: scores.append(100)
    elif cod <= 20: scores.append(70)
    elif cod <= 50: scores.append(40)
    else: scores.append(10)
        
    # DO
    do = row['DO(mg/l)']
    if do >= 6: scores.append(100)
    elif do >= 5: scores.append(80)
    elif do >= 4: scores.append(50)
    else: scores.append(10)

    # pH
    ph = row['pH']
    if 6.5 <= ph <= 8.5: scores.append(100)
    elif 6.0 <= ph < 6.5 or 8.5 < ph <= 9.0: scores.append(60)
    else: scores.append(20)

    return np.mean(scores)

def score_to_class(score):
    if score >= 80: return 0  
    elif score >= 60: return 1  
    elif score >= 40: return 2  
    else: return 3  

print("\nCalculating quality scores...")
df['quality_score'] = df.apply(calculate_river_quality, axis=1)
df['quality_label'] = df['quality_score'].apply(score_to_class)

# 8. Create Final Dataset 
# The hardware sheet has a Red X for COD, so we drop it.
# We also drop the raw quality_score.
cols_to_drop = ['COD(mg/l)', 'quality_score']
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df_final = df.drop(columns=existing_cols_to_drop)

# 9. Save to CSV
output_path = r"C:\Users\omen\OneDrive\Documents\wq_data_hp\cleaned_beas_data.csv"
df_final.to_csv(output_path, index=False)
print(f"\n Final cleaned dataset saved to: {output_path}")
print(f"Columns in final dataset: {list(df_final.columns)}")
