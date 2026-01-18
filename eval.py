import pandas as pd

# Paths (assumes running from project root)
results_path = "resultsSTORY.csv"
train_path = "train.csv"

# Read files
res = pd.read_csv(results_path)
train = pd.read_csv(train_path)

# Normalize column names
res.columns = [c.strip() for c in res.columns]
train.columns = [c.strip() for c in train.columns]

# Determine ID column names
res_id_col = None
for c in ["Story ID", "StoryID", "id"]:
    if c in res.columns:
        res_id_col = c
        break
if res_id_col is None:
    raise ValueError(f"No ID column found in {results_path}; columns: {res.columns.tolist()}")

train_id_col = None
for c in ["id", "ID", "Id"]:
    if c in train.columns:
        train_id_col = c
        break
if train_id_col is None:
    raise ValueError(f"No ID column found in {train_path}; columns: {train.columns.tolist()}")

# Ensure prediction column exists
pred_col = None
for c in ["Prediction", "prediction", "pred"]:
    if c in res.columns:
        pred_col = c
        break
if pred_col is None:
    raise ValueError(f"No Prediction column found in {results_path}; columns: {res.columns.tolist()}")

# Map train labels to numeric (consistent -> 1, contradict -> 0)
if "label" not in train.columns:
    raise ValueError(f"No 'label' column found in {train_path}; columns: {train.columns.tolist()}")

label_map = {"consistent": 1, "contradict": 0}
train = train.copy()
train["label_numeric"] = train["label"].astype(str).str.lower().map(label_map)

# Convert ID types and merge
res[res_id_col] = res[res_id_col].astype(str).str.strip().astype(int)
train[train_id_col] = train[train_id_col].astype(int)

merged = pd.merge(res, train, left_on=res_id_col, right_on=train_id_col, how="inner")

if merged.empty:
    print("No matching IDs between the two files.")
    raise SystemExit(1)

# Ensure predictions are numeric
merged[pred_col] = pd.to_numeric(merged[pred_col], errors="coerce")

# Drop rows with NaN predictions or labels
before = len(merged)
merged = merged.dropna(subset=[pred_col, "label_numeric"]) 
after = len(merged)
if after < before:
    print(f"Dropped {before-after} rows due to NaN in prediction or label_numeric.")

# Compute accuracy
merged["correct"] = (merged[pred_col].astype(int) == merged["label_numeric"].astype(int))
correct = merged["correct"].sum()
total = len(merged)
accuracy = correct / total

print("Evaluation summary")
print("---")
print(f"Matched examples: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Print confusion details
tp = ((merged[pred_col] == 1) & (merged["label_numeric"] == 1)).sum()
tn = ((merged[pred_col] == 0) & (merged["label_numeric"] == 0)).sum()
fp = ((merged[pred_col] == 1) & (merged["label_numeric"] == 0)).sum()
fn = ((merged[pred_col] == 0) & (merged["label_numeric"] == 1)).sum()
print("---")
print(f"True Positives (pred=1,label=1): {tp}")
print(f"True Negatives (pred=0,label=0): {tn}")
print(f"False Positives (pred=1,label=0): {fp}")
print(f"False Negatives (pred=0,label=1): {fn}")

# Show up to 10 mismatches
mismatches = merged[merged["correct"] == False]
if not mismatches.empty:
    print("---")
    print("Sample mismatches (up to 10):")
    print(mismatches[[res_id_col, pred_col, "label", "label_numeric", "Rationale"]].head(10).to_string(index=False))

# Exit code 0
