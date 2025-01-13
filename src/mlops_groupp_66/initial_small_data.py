import pandas as pd

# File path
file_path = r'D:\VScode\Projects\MLOps_Final_work\creditcard.csv'

# Load the original CSV file
df = pd.read_csv(file_path)

# Filter the dataset to separate class 0 and class 1
class_0 = df[df['Class'] == 0]
class_1 = df[df['Class'] == 1]

# Randomly sample 508 observations from class 0 and 492 from class 1
class_0_sampled = class_0.sample(n=508, random_state=42)
class_1_sampled = class_1.sample(n=492, random_state=42)

# Combine the sampled data
balanced_df = pd.concat([class_0_sampled, class_1_sampled])

# Shuffle the combined dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = r'D:\VScode\Projects\MLOps_Final_work\balanced_creditcard.csv'
balanced_df.to_csv(output_path, index=False)

print("New CSV file created with 508 class 0 and 492 class 1 observations.")

# Count the observations in each class
class_counts = balanced_df['Class'].value_counts()
print("Class counts in the new dataset:")
print(class_counts)