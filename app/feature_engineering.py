import pandas as pd


def create_features(df):

    df = df.sort_values(
        ['State', 'Date']
    )

    # Lag Features
    df['lag_1'] = (
        df.groupby('State')['Total']
        .shift(1)
    )

    df['lag_7'] = (
        df.groupby('State')['Total']
        .shift(7)
    )

    df['lag_30'] = (
        df.groupby('State')['Total']
        .shift(30)
    )

    # Rolling Mean
    df['rolling_mean_7'] = (
        df.groupby('State')['Total']
        .transform(
            lambda x: x.rolling(7).mean()
        )
    )

    # Rolling Std
    df['rolling_std_7'] = (
        df.groupby('State')['Total']
        .transform(
            lambda x: x.rolling(7).std()
        )
    )

    # Date Features
    df['month'] = df['Date'].dt.month

    df['week'] = (
        df['Date']
        .dt.isocalendar()
        .week.astype(int)
    )

    df['year'] = df['Date'].dt.year

    # Holiday Flag
    df['holiday_flag'] = df['month'].apply(
        lambda x: 1 if x in [11, 12] else 0
    )

    # Remove nulls
    df = df.dropna()

    return df