# state_impute.py
import pandas as pd
import numpy as np
import re
import logging
from config import US_STATES, US_STATES_FULL



class StateImputer:

    @staticmethod
    def correct_short_zipcodes_fast_safe(df, mapping_path):
        """Step 5: Fix short zipcodes using mapping file"""
        import re
        mapping_df = pd.read_excel(mapping_path, dtype={'zipcode': str, 'zipcode3': str, 'zipcode4': str})
        mapping_df['zipcode'] = mapping_df['zipcode'].astype(str)
        mapping_df['zipcode3'] = mapping_df['zipcode3'].astype(str)
        mapping_df['zipcode4'] = mapping_df['zipcode4'].astype(str)
        df = df.copy()
        df['zipcode'] = df['zipcode'].astype(str)
        df['original_zipcode'] = df['zipcode']
        # Map 3-digit
        mapping3 = mapping_df[['zipcode3', 'zipcode']].dropna().drop_duplicates(subset=['zipcode3'])
        map3 = mapping3.set_index('zipcode3')['zipcode']
        mask3 = df['zipcode'].str.len() == 3
        df.loc[mask3, 'correct_zipcode'] = df.loc[mask3, 'zipcode'].map(map3)
        # Map 4-digit
        mapping4 = mapping_df[['zipcode4', 'zipcode']].dropna().drop_duplicates(subset=['zipcode4'])
        map4 = mapping4.set_index('zipcode4')['zipcode']
        mask4 = df['zipcode'].str.len() == 4
        df.loc[mask4, 'correct_zipcode'] = df.loc[mask4, 'zipcode'].map(map4)
        def is_us_zip(val):
            return bool(re.fullmatch(r'\d{5}', str(val)))
        df['zipcode'] = df.apply(
            lambda row: row['correct_zipcode'] if is_us_zip(row.get('correct_zipcode', '')) else row['zipcode'],
            axis=1
        )
        df = df.drop(columns=['correct_zipcode'])
        logging.info(f"Short zipcodes corrected using mapping. Now {df['zipcode'].str.len().value_counts().to_dict()} by length.")
        return df
    
    @staticmethod
    def get_zipcode_to_state(path):
        """Step 1: Build zipcode -> state mapping from excel"""
        canaria_map = pd.read_excel(path, dtype={'zipcode': str, 'state': str})
        canaria_map['zipcode'] = canaria_map['zipcode'].str.zfill(5)
        return dict(zip(canaria_map['zipcode'], canaria_map['state']))

    @staticmethod
    def impute_states(df, zipcode_to_state, US_STATES=US_STATES, US_STATES_FULL=US_STATES_FULL):
        """Step 2: Impute missing states based on rules"""
        df = df.copy()
        df['correct_state'] = df['state'].where(df['state'].isin(US_STATES), np.nan)
        remote_mask = df['correct_state'].isna() & (df['zipcode'].astype(str).str.strip().str.lower() == 'remote')
        df.loc[remote_mask, 'correct_state'] = 'REMOTE'
        zipcode_mask = df['correct_state'].isna()
        zipcodes_padded = df.loc[zipcode_mask, 'zipcode'].astype(str).str.zfill(5)
        df.loc[zipcode_mask, 'correct_state'] = zipcodes_padded.map(zipcode_to_state)
        # From location
        state_codes = list(US_STATES)
        state_pattern = r'\b(' + '|'.join(state_codes) + r')\b'
        df['state_from_location'] = df['scraped_location'].str.extract(state_pattern, expand=False)
        null_state_mask = df['correct_state'].isna() | (df['correct_state'] == '')
        df.loc[null_state_mask & df['state_from_location'].notna(), 'correct_state'] = df.loc[null_state_mask & df['state_from_location'].notna(), 'state_from_location']
        # From full state names
        state_full_to_abbr = {}
        for abbr, full in US_STATES_FULL.items():
            for variant in [full.upper(), full.replace(' ', '').upper(), full.title(), full.replace(' ', '').title(), full.lower(), full.replace(' ', '').lower(), full.replace('-', ' ').title(), full.replace('-', '').title()]:
                state_full_to_abbr[variant] = abbr
        state_names_pattern = '|'.join(sorted(state_full_to_abbr.keys(), key=len, reverse=True))
        pattern = re.compile(r'\b(' + state_names_pattern + r')\b', re.IGNORECASE)
        def extract_state_from_location(location):
            if pd.isna(location): return np.nan
            match = pattern.search(location)
            if match: return state_full_to_abbr.get(match.group(0).strip().replace(' ', '').upper())
            return np.nan
        mask = df['correct_state'].isna()
        df.loc[mask, 'correct_state'] = df.loc[mask, 'scraped_location'].apply(extract_state_from_location)
        if 'state_from_location' in df.columns:
            df = df.drop(columns=['state_from_location'])
        return df

    @staticmethod
    def assign_us_geo_level(df):
        """Step 3: Assign US as geo_level for all rows"""
        df = df.copy()
        df['geo_level'] = 'US'
        return df
    
    @staticmethod
    def print_date_coverage(
        df,
        date_col='correct_date_filled',
        geo_col='geo_level',
        src_col='src',
        max_groups=5
    ):
        """
        Prints/logs date range and count of records for each (geo_level, src) or (state, src) combo.
        Useful for data sanity checking before anomaly detection.
        """
        df_clean = df.dropna(subset=[geo_col, src_col])
        groupby_cols = [geo_col, src_col]
        groups = df_clean.groupby(groupby_cols)

        for idx, ((geo, src), group) in enumerate(groups):
            min_date = group[date_col].min()
            max_date = group[date_col].max()
            total = len(group)
            null_count = group[date_col].isnull().sum()
            logging.info(
                f"({geo}, {src}) — Records: {total} | Date Range: {min_date} to {max_date} | Nulls: {null_count}"
            )
            if idx + 1 >= max_groups:
                break

        logging.info(f"Logged date coverage for {min(max_groups, len(groups))} ({geo_col}, {src_col}) combos.")
    
    @staticmethod
    def check_missing_combinations(
    df: pd.DataFrame,
    date_col: str = 'correct_date_filled',
    geo_col: str = 'geo_level',
    src_col: str = 'src',
    valid_geos: set = None,
    master_sources: set = None
    ):
        """
        Logs any (geo_level, src) combinations missing for the latest day,
        only considering valid geos if provided.
        """
        df[date_col] = pd.to_datetime(df[date_col])
        today = df[date_col].max().date()
        logging.info(f"Checking missing {geo_col} x {src_col} combinations for {today}...")

        # Only include valid geo levels, if provided
        if valid_geos is not None:
            geos = set(df[geo_col].dropna().unique()) & set(valid_geos)
        else:
            geos = set(df[geo_col].dropna().unique())

        if master_sources is not None:
            sources = set(master_sources)
        else:
            sources = set(df[src_col].dropna().unique())

        expected_combinations = {(g, src) for g in geos for src in sources}
        df_today = df[df[date_col].dt.date == today]
        present_combinations = set(zip(df_today[geo_col], df_today[src_col]))
        missing_combinations = expected_combinations - present_combinations

        if missing_combinations:
            logging.warning(f"Missing {geo_col} x {src_col} combos for {today}: {sorted(missing_combinations)}")
        else:
            logging.info(f"All {geo_col} x {src_col} combos present for {today}.")




