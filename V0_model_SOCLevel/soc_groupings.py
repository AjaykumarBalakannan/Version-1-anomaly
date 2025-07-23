import pandas as pd

def is_seasonal_burst(series, spike_pct=6.0, lull_thresh=1000, min_lull_days=14, min_spikes=2):
    """
    Returns True if there are at least min_spikes in the entire series,
    and each spike is followed by a lull (below lull_thresh) of at least min_lull_days.
    """
    mean = series.mean()
    std = series.std()
    spike_thresh = std * spike_pct

    # Find all spike days in the entire series
    spike_days = series[series >= spike_thresh].index.tolist()
    valid_spikes = 0

    for spike_day in spike_days:
        after_spike = series.loc[spike_day:]
        lull_days = (after_spike < lull_thresh).sum()
        if lull_days >= min_lull_days:
            valid_spikes += 1

    return valid_spikes >= min_spikes

def rollup_soc_code(code, level=1):
    """
    Roll up SOC code to parent grouping.
    level=1: roll only last digit to zero (e.g. 11-1011 -> 11-1010)
    level=2: roll last two digits to zero (e.g. 11-1011/11-1010 -> 11-1000)
    """
    if not isinstance(code, str) or len(code) != 7 or '-' not in code:
        return code  # handle malformed
    head, tail = code.split('-')
    if level == 1:
        return f"{head}-{tail[:-1]}0"
    elif level == 2:
        return f"{head}-{tail[0]}000"
    else:
        return code

# Example usage (to be used in pipeline):
# mean_counts = daily_counts.groupby('nlp_soc_code')['job_count'].mean()
# low_volume_6d = mean_counts[mean_counts < 50].index
# daily_counts['soc_code_rolled'] = daily_counts['nlp_soc_code'].apply(
#     lambda x: rollup_soc_code(x, level=1) if x in low_volume_6d else x
# )
# mean_counts_5d = daily_counts.groupby('soc_code_rolled')['job_count'].mean()
# low_volume_5d = mean_counts_5d[mean_counts_5d < 50].index 