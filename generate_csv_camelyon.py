import os
import pandas as pd

# === CONFIGURATION ===
train_tumor_dir = "/vol/research/datasets/pathology/Camelyon/Camelyon16/training/tumor"
train_normal_dir = "/vol/research/datasets/pathology/Camelyon/Camelyon16/training/normal"
test_dir = "/vol/research/datasets/pathology/Camelyon/Camelyon16/testing/images"
reference_csv_path = "/vol/research/datasets/pathology/Camelyon/Camelyon16/testing/reference.csv"
output_path = "dataset_csv/camelyon16.csv"

# === Load reference.csv for test labels ===
reference_df = pd.read_csv(reference_csv_path, header=None)
reference_df.columns = ['slide', 'cancer', 'subtype', 'level']
reference_df['slide'] = (
    reference_df['slide']
    .astype(str)
    .str.encode('ascii', 'ignore')
    .str.decode('ascii')
    .str.strip()
)
reference_df['slide_id'] = reference_df['slide'].apply(lambda x: os.path.splitext(x)[0])
reference_df['label'] = reference_df['cancer'].map({'Tumor': 1, 'Normal': 0})
reference_label_map = dict(zip(reference_df['slide_id'], reference_df['label']))

def collect_slide_info(folder, label, split):
    rows = []
    for fname in os.listdir(folder):
        if not fname.endswith(('.tif', '.svs')):
            continue
        base_id = os.path.splitext(fname)[0]
        rows.append({
            "slide_id": base_id,
            "full_path": os.path.join(folder, fname),
            "label": label,
            "split": split
        })
    return rows

# === Training entries from folder structure ===
entries = []
entries += collect_slide_info(train_tumor_dir, label=1, split="train")
entries += collect_slide_info(train_normal_dir, label=0, split="train")

# === Test entries with label from reference.csv ===
test_entries = []
for fname in os.listdir(test_dir):
    if not fname.endswith(('.tif', '.svs')):
        continue
    base_id = os.path.splitext(fname)[0]
    test_label = reference_label_map.get(base_id, -1)  # fallback -1 if not found
    test_entries.append({
        "slide_id": base_id,
        "full_path": os.path.join(test_dir, fname),
        "label": test_label,
        "split": "test"
    })

entries += test_entries

# === Save final DataFrame ===
df = pd.DataFrame(entries)
df = df[['slide_id', 'full_path', 'label', 'split']]
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

# === Confirm Output ===
print(f"✅ camelyon16.csv created: {output_path}")
print(f"🗂️ Total slides: {len(df)}")
print("🔬 Train label breakdown:", df[df['split'] == 'train']['label'].value_counts().to_dict())
print("🧪 Test label breakdown:", df[df['split'] == 'test']['label'].value_counts().to_dict())
print("\n🖼️ First 5 test entries:")
print(df[df['split'] == 'test'].head(5).to_string(index=False))
