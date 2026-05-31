import json
import hashlib

def sample_hash(sample):
    """
    Create a stable hash for a sample.
    Works even if key ordering differs.
    """
    sample_str = json.dumps(sample, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(sample_str.encode("utf-8")).hexdigest()

def load_hashes(jsonl_path):
    hashes = set()
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            hashes.add(sample_hash(sample))
            total += 1

    return hashes, total

train_file = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_10000.json"
val_file = "Dataset/ApolloCorpus/pretrain/medicalGuideline_en_qa_2000_val.json"

print("Loading train dataset...")
train_hashes, train_total = load_hashes(train_file)

print("Loading validation dataset...")
val_hashes, val_total = load_hashes(val_file)

overlap = train_hashes.intersection(val_hashes)

print("\n===== RESULTS =====")
print(f"Train samples      : {train_total}")
print(f"Validation samples : {val_total}")
print(f"Exact overlaps     : {len(overlap)}")

if val_total > 0:
    print(f"Overlap % of validation: {100 * len(overlap) / val_total:.2f}%")

if train_total > 0:
    print(f"Overlap % of training  : {100 * len(overlap) / train_total:.2f}%")

if len(overlap) == 0:
    print("\n✅ No exact duplicates found.")
else:
    print(f"\n⚠️ ALERT: Found {len(overlap)} OVERLAPPING SAMPLES!")