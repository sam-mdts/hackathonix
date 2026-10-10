import sys
from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, LeaveOneGroupOut

df = pd.read_csv("all_data.csv")
X = df.drop(columns=["person", "label"]).values
y = df["label"].values
groups = df["person"].values
people = sorted(set(groups))

print(f"Rows: {len(df)} | People: {len(people)} | Letters: {len(set(y))}")
if len(people) < 2:
    print("Need at least 2 people to test on someone unseen. Collect more data.")
    sys.exit()

def make_model():
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)

# 1) Random split (the "easy" test that mixes everyone together)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)
random_acc = (make_model().fit(Xtr, ytr).predict(Xte) == yte).mean()
print(f"\nRandom split accuracy: {random_acc:.1%}")

# 2) Leave-one-person-out (the honest test on someone never seen)
per_person = {}
confusions = Counter()
for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
    held_out = groups[test_idx][0]
    model = make_model().fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    per_person[held_out] = (pred == y[test_idx]).mean()
    for true, guess in zip(y[test_idx], pred):
        if true != guess:
            confusions[(true, guess)] += 1

print("\nAccuracy on each unseen person:")
for person, acc in per_person.items():
    print(f"  {person}: {acc:.1%}")
mean_acc = sum(per_person.values()) / len(per_person)
print(f"\nMean unseen-person accuracy: {mean_acc:.1%}")
print(f"Gap vs random split: {random_acc - mean_acc:.1%}")

print("\nMost confused letters (true -> guessed, count):")
for (true, guess), n in confusions.most_common(8):
    print(f"  {true} -> {guess}: {n}")
