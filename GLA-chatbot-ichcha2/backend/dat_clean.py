import pandas as pd
import os
import glob
import json

# Step 1: Paths
base_folder = "D:/gla2 chatbot/GLA-chatbot/cleaned_data/Institute"  # Main folder
existing_json_path = "D:/gla2 chatbot/GLA-chatbot/cleaned_data/gla_cleaned.json"
output_json = 'cleaned_data/combined_data.json'
os.makedirs('cleaned_data', exist_ok=True)

# Step 2: Load CSVs from Institute and Department
csv_paths = glob.glob(os.path.join(base_folder, 'Institute', '*.csv')) + \
            glob.glob(os.path.join(base_folder, 'Department', '*.csv'))

all_dataframes = []

for path in csv_paths:
    try:
        df = pd.read_csv(path)
        df.dropna(how='all', inplace=True)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()

        df['source_file'] = os.path.basename(path)
        df['source_folder'] = os.path.basename(os.path.dirname(path))
        all_dataframes.append(df)

    except Exception as e:
        print(f"❌ Error reading {path}: {e}")

# Step 3: Load JSON files from Courses/
json_paths = glob.glob(os.path.join(base_folder, 'Courses', '*.json'))
for path in json_paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        course_df = pd.DataFrame(data)

        # Clean it
        course_df.columns = [col.strip().lower().replace(' ', '_') for col in course_df.columns]
        for col in course_df.select_dtypes(include='object').columns:
            course_df[col] = course_df[col].astype(str).str.strip()

        course_df['source_file'] = os.path.basename(path)
        course_df['source_folder'] = 'Courses'

        all_dataframes.append(course_df)

    except Exception as e:
        print(f"❌ Error reading JSON {path}: {e}")

# Step 4: Combine CSV + JSON (Courses)
combined_df = pd.concat(all_dataframes, ignore_index=True)

# Step 5: Load your already cleaned 1–8 JSON data
if os.path.exists(existing_json_path):
    with open(existing_json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    existing_df = pd.DataFrame(json_data)
else:
    print("⚠️ Could not find existing cleaned JSON file.")
    existing_df = pd.DataFrame()

# Step 6: Merge everything
final_df = pd.concat([existing_df, combined_df], ignore_index=True)

# Step 7: Save final combined JSON
final_df.to_json(output_json, orient='records', indent=2)
print(f"✅ Final combined JSON saved to {output_json}")
