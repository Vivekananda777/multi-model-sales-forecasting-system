import pandas as pd


def load_data(path):

    df = pd.read_excel(path)

    print(df.columns)

    df['Date'] = pd.to_datetime(df['Date'])

    df = (
        df.groupby(['State', 'Date'])['Total']
        .sum()
        .reset_index()
    )

    return df


def handle_missing_dates(df):

    final_df = []

    for state in df['State'].unique():

        temp = df[df['State'] == state].copy()

        temp = temp.set_index('Date')

        full_dates = pd.date_range(
            start=temp.index.min(),
            end=temp.index.max(),
            freq='W'
        )

        temp = temp.reindex(full_dates)

        temp['State'] = state

        temp['Total'] = temp['Total'].interpolate()

        temp = temp.reset_index()

        temp.rename(columns={'index': 'Date'}, inplace=True)

        final_df.append(temp)

    final_df = pd.concat(final_df)

    return final_df