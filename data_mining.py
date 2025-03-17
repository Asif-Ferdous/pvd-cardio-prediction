import pandas as pd

# Load the dataset
file_path = "C:/Users/m_m_a/Music/asifT/updated_cardio_train_with_PVD.csv"
df = pd.read_csv(file_path)

# Convert 'age' from days to years
df["age"] = df["age"] // 365  # Convert to whole years

# Save the updated dataset
updated_file_path = "C:/Users/m_m_a/Music/asifT/converted_cardio_train_with_PVD_years.csv"
df.to_csv(updated_file_path, index=False)

print(f"✅ Age converted to years and saved as {updated_file_path}")
