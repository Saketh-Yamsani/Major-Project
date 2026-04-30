import pandas as pd
import json
import os

# 🔹 Step 1: Give your JSON file path here
json_file = r"C:\\Users\\5sake\Downloads\\bulb_flood_attack.json"   # <-- change this

# 🔹 Step 2: Load JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 🔹 Step 3: Convert JSON to DataFrame
df = pd.json_normalize(data)

# 🔹 Step 4: Create CSV file name automatically
csv_file = os.path.splitext(json_file)[0] + ".csv"

# 🔹 Step 5: Save CSV
df.to_csv(csv_file, index=False)

print("✅ Conversion completed!")
print("CSV file created:", csv_file)
