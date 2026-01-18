
import pandas as pd

# Load test data
cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
test_df = pd.read_csv('test_FD001.txt/test_FD001.txt', sep=r'\s+', header=None, names=cols)

# Check lengths
short_units = []
for unit_id in test_df['unit'].unique():
    unit_len = len(test_df[test_df['unit'] == unit_id])
    if unit_len < 30:
        short_units.append((unit_id, unit_len))

print(f"Total Units: {len(test_df['unit'].unique())}")
print(f"First 10 Unit IDs: {test_df['unit'].unique()[:10]}")
print(f"Short Units (<30 cycles): {short_units}")
if short_units:
    print("These units are being skipped by the current logic.")
