import pandas as pd

# Load a sample or full batch from your parquet file (adjust path as needed)
batch_df = pd.read_parquet('data/linkedin_jan_2025.parquet')

# Ensure roll-up logic is available
from soc_groupings import rollup_soc_code

# Apply roll-up (same as in pipeline)
if 'nlp_soc_code' in batch_df.columns:
    mean_counts = batch_df.groupby('nlp_soc_code')['job_count'].mean()
    mean_counts = pd.Series(mean_counts, index=batch_df['nlp_soc_code'].unique())
    low_volume_6d = mean_counts[mean_counts < 50].index
    batch_df['nlp_soc_code_rolled'] = batch_df['nlp_soc_code'].apply(
        lambda x: rollup_soc_code(x, level=1) if x in low_volume_6d else x
    )
else:
    raise ValueError("'nlp_soc_code' column not found in input data.")

# Before roll-up: sum job counts by original code
before_rollup = batch_df.groupby('nlp_soc_code')['job_count'].sum().reset_index()
before_rollup.columns = ['nlp_soc_code', 'job_count_before']

# After roll-up: sum job counts by rolled code
after_rollup = batch_df.groupby('nlp_soc_code_rolled')['job_count'].sum().reset_index()
after_rollup.columns = ['nlp_soc_code_rolled', 'job_count_after']

# Merge to compare
comparison = before_rollup.merge(
    batch_df[['nlp_soc_code', 'nlp_soc_code_rolled']].drop_duplicates(),
    on='nlp_soc_code',
    how='left'
).merge(
    after_rollup,
    on='nlp_soc_code_rolled',
    how='left'
)

print(comparison[['nlp_soc_code', 'nlp_soc_code_rolled', 'job_count_before', 'job_count_after']].head(50)) 