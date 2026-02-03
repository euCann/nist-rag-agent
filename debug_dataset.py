#!/usr/bin/env python3
"""Debug script to inspect NIST dataset structure."""

from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset(
    "ethanolivertroy/nist-cybersecurity-training",
    split="train"
)

print(f"\nDataset size: {len(dataset)}")
print(f"\nDataset features: {dataset.features}")
print(f"\nDataset column names: {dataset.column_names}")

# Print first example
if len(dataset) > 0:
    print("\n=== First Example ===")
    first = dataset[0]
    for key, value in first.items():
        print(f"\n{key}:")
        if isinstance(value, str):
            print(f"  {value[:200]}..." if len(value) > 200 else f"  {value}")
        else:
            print(f"  {value}")
