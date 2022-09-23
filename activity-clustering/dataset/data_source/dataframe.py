"""
Get formatted Nagios CPU time series dataframes
"""
import pandas as pd

def interval_cpu_mean_df(df, interval_seconds):
    """
    Get a time series of CPU averages on the specified time interval

    df: input dataframe, having ('date', 'hostname', 'os', 'cpu') format
    interval_seconds: time series interval in seconds

    Return a dataframe of format:
    (date, average_cpu)
    """
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_date_cpu = df_copy[['date', 'cpu']].groupby(
        pd.Grouper(
            key='date',
            freq='{}S'.format(interval_seconds)
        )
    )
    return df_date_cpu.mean()

def auto_interval_cpu_mean_df(df, max_rows):
    """
    Get a time series of CPU averages with interval automatically set to
    fit the specified max number of rows

    df: input dataframe, having ('date', 'hostname', 'os', 'cpu') format
    max_rows: maximum number of rows in output dataframe

    Return a (output_df, interval_seconds, interval_str) tuple
    output_df: (date, average_cpu) dataframe
    interval_seconds: interval in seconds
    interval_str: interval in the form '<n>day, <n>hour, <n>minute'
    """
    time_span_seconds = (df['date'].iloc[-1] - df['date'].iloc[0]).total_seconds()
    interval_seconds = int(time_span_seconds / (max_rows - 1))
    interval_seconds = interval_seconds or 1 # In case all records on same day-hour-minute

    # Format interval display to numbers of days/hours/minutes
    days = interval_seconds // (60 * 60 * 24)
    seconds = interval_seconds - (days * 60 * 60 * 24)
    hours = seconds // (60 * 60)
    seconds = seconds - hours * 60 * 60
    minutes = seconds // 60
    interval_str = '{}day {}hour {}minute'.format(int(days), int(hours), int(minutes))

    df_date_cpu = interval_cpu_mean_df(df, interval_seconds)
    return (df_date_cpu, interval_seconds, interval_str)
