# debug_check_data.py
from .storage.npz_storage import NPZReader

reader = NPZReader()
df = reader.read_features()

print(f"Total records: {len(df)}")
print(f"Unique segments: {df['segment_id'].nunique()}")

if 'date_range' in df.columns:
    print(f"Unique dates: {df['date_range'].nunique()}")
    print(f"Date range values: {sorted(df['date_range'].unique())[:5]}...")  # First 5 dates

if 'time_set' in df.columns:
    print(f"Unique time slots: {df['time_set'].nunique()}")

# Check one segment
seg_id = df['segment_id'].iloc[0]
seg_df = df[df['segment_id'] == seg_id]

print(f"\nSegment {seg_id}:")
print(f"  Total records: {len(seg_df)}")

if 'date_range' in df.columns:
    print(f"  Unique dates: {seg_df['date_range'].nunique()}")
if 'time_set' in df.columns:
    print(f"  Unique time slots: {seg_df['time_set'].nunique()}")

# Expected: 31 days × 24 slots = 744 records per segment
print(f"\n Expected: 31 × 24 = 744 records per segment")
print(f" Actual: {len(seg_df)} records")