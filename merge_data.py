import glob, os, sys
import pandas as pd

sources = {}
if os.path.exists("data.csv"):
    sources["data.csv (this laptop)"] = pd.read_csv("data.csv")
for path in sorted(glob.glob("parts/*.csv")):
    sources[path] = pd.read_csv(path)

if not sources:
    print("No data files found.")
    sys.exit()

# Make sure no person ID shows up in two different files
seen = {}
problem = False
for name, df in sources.items():
    for pid in df["person"].unique():
        if pid in seen:
            print(f"PROBLEM: ID {pid} is in both {seen[pid]} and {name}")
            problem = True
        seen[pid] = name
if problem:
    print("Fix the IDs before merging. Nothing was written.")
    sys.exit()

combined = pd.concat(sources.values(), ignore_index=True)
combined.to_csv("all_data.csv", index=False)

print(f"Saved all_data.csv with {len(combined)} rows.\n")
print("Per person:")
for pid, group in combined.groupby("person"):
    print(f"  {pid}: {len(group)} rows, {group['label'].nunique()} letters")