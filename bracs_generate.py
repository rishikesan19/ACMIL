import os
import pandas as pd

# ---------------- Configuration ----------------
base_path = "/vol/research/scratch1/NOBACKUP/rk01337/BRACS/histoimage.na.icar.cnr.it/BRACS_WSI"
output_csv = "./dataset_csv/bracs.csv"  # adjusted path

# Label mapping
label_mapping = {
    "Type_ADH": 4,
    "Type_DCIS": 5,
    "Type_FEA": 3,
    "Type_IC": 6,
    "Type_N": 0,
    "Type_PB": 1,
    "Type_UDH": 2
}

# Ensure output folder exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# ---------------- Processing ----------------
data = []
patient_id = 0

for split in ["train", "val", "test"]:
    split_path = os.path.join(base_path, split)
    for root, dirs, files in os.walk(split_path):
        for file in files:
            if file.endswith(".svs"):
                full_path = os.path.join(root, file)
                slide_id = os.path.splitext(file)[0]
                case_id = f"patient_{patient_id}"
                patient_id += 1

                # Extract Type_X for label
                try:
                    type_folder = root.split("/")[-1]
                    label = label_mapping.get(type_folder, -1)
                    if label == -1:
                        print(f"⚠️ Warning: Unrecognized type folder '{type_folder}' in {full_path}")
                except Exception as e:
                    print(f"❌ Error extracting label from {full_path}: {e}")
                    label = -1

                data.append({
                    "case_id": case_id,
                    "slide_id": slide_id,
                    "split_info": split,
                    "full_path": full_path,
                    "label": label
                })

# ---------------- Saving CSV ----------------
df = pd.DataFrame(data)
df.reset_index(inplace=True)  # adds 'Unnamed: 0' column
df.to_csv(output_csv, index=False)

print(f"✅ CSV generated: {output_csv} with {len(df)} entries.")
print(df.head())
