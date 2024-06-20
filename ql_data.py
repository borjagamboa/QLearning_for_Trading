from ib_api import IBapi
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import os, glob
import time, threading
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def run_loop(app):
    app.run()

def get_data(ticker, new=True):
    path_to_folder = os.path.join(os.path.join(os.getcwd(), 'Data'), ticker)
    if new:
        app = IBapi()
        app.connect("127.0.0.1", 7497, clientId=123)

        # Start the socket in a thread
        api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
        api_thread.start()
        time.sleep(1) # Sleep interval to allow time for connection to server
        print("serverVersion:%s connectionTime:%s" % (app.serverVersion(), app.twsConnectionTime()))
        try:
            df = app.get_cripto_historical(ticker)
            dt_time = datetime.now().strftime('%y-%m-%d-%H%M')
            if path_to_folder:
                pass
            else:
                os.mkdir(ticker)
            df.to_csv(f'{ticker}/{dt_time}.csv')
            print(f'New data saved into {ticker}/{dt_time}.csv')
            app.disconnect()
            return df
        except:
            print('Unable to save data')
            app.disconnect()
            return []
    else:
        if not glob.glob(os.path.join(path_to_folder, '*')):
            print("Folder is empty")
            return []
        else:
            path_to_file = glob.glob(os.path.join(path_to_folder, '*'))[-1]
            print(path_to_file)
            data = pd.read_csv(path_to_file, index_col='DateTime')
            return data

def bollinger_bands(series: pd.Series, length: int = 20, *, num_stds: tuple[float, ...] = (2, 0, -2),
                    prefix: str = '') -> pd.DataFrame:
    # Ref: https://stackoverflow.com/a/74283044/
    rolling = series.rolling(length)
    bband0 = rolling.mean()
    bband_std = rolling.std(ddof=0)
    return pd.DataFrame({f'{prefix}{num_std}': (bband0 + (bband_std * num_std)) for num_std in num_stds})

def get_delta_avg(series: pd.Series, period: int = 20, prefix: str = 'delta_avg'):
    delta = series.pct_change(1)
    return pd.DataFrame({f'{prefix}_{period}': delta.rolling(period).mean()})

def aggregate_indicator(data, column, indicator, length=20, num_stds=(2, 0, -2), prefix='BB_', signal=True):
    '''
    Aggregate defined indicators to DataFrame.
        - data: the indicator will be calculated and aggregate within this DataFrame
        - column: column on which the indicator will be calculated
        - length: period for which the indicator will be calculated, in case needed
        - num_stds: number of stds to take in account for 'bollinger bands'
        - signal: if True, a column will be added showing a trading signal provided the indicator allows it
    '''
    if indicator == 'bollinger':
        b_bands = bollinger_bands(data[column], length, num_stds=num_stds, prefix=prefix)
        data = data.join(b_bands, on=['DateTime']).iloc[length:]
        if signal:
            conditions = [data[column] > data[f'{prefix}{num_stds[0]}'],
                          data[column] < data[f'{prefix}{num_stds[2]}']]
            choices = ['Overvalued', 'Undervalued']  # -1.0 = undervalued, 1.0=overvalued, 0.0=marketprice
            data['Signal'] = np.select(conditions, choices, 'Market')
        else:
            pass
    if indicator == 'delta_avg':
        '''
        Average unit percentage increase for a period
        '''
        delta_avg = get_delta_avg(data[column], length, prefix=prefix)
        data = data.join(delta_avg, on=['DateTime']).iloc[length:]
    return data

def cat_to_dummies(data, column, dropkey: str=None):
    '''
    Get dummies from categorical column and drop it
    Use dropkey to drop a category
    '''
    # Using OneHotEncoder
    if dropkey is not None:
        drop = [dropkey]
    encoder = OneHotEncoder(drop=drop, sparse_output=False, handle_unknown='error')
    X_encoded = pd.DataFrame(encoder.fit_transform(data[[column]]), columns=encoder.get_feature_names_out())
    X_encoded['DateTime'] = data.index
    X_encoded.set_index(['DateTime'], inplace=True)
    return data.join(X_encoded, on=['DateTime']).drop([column], axis=1)

def aggregate_categories(data, columns, n_clusters):
    '''
    Aggregates categories based on K-Means method and by using the desired columns.
        - data: Dataframe in which the categories will be calculated
        - columns: desired columns to take in account
        - n_cluster: number of clusters to define
    '''
    # Take only desired columns
    cluster_data = data[columns]

    # Convert DataFrame to matrix
    mat = cluster_data.values
    # Using sklearn
    km = KMeans(n_clusters=n_clusters)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([data.index, labels]).T
    results.columns = ['DateTime', 'Category']
    results.set_index('DateTime', inplace=True)
    results = pd.to_numeric(results['Category'])
    data = data.join(results, on=['DateTime'])
    return data

def plot_results(data):
    fig, ax = plt.subplots(dpi=300, figsize=(10, 3))
    plt_data = data['2024-04-20 16:00:00':'2024-04-23 16:00:00']
    plt_data['Close'].plot(ax=ax, lw=0.5)
    plt_data['CloseBB5_1.5'].plot(ax=ax, ls='--', lw=0.5)
    plt_data['CloseBB5_0'].plot(ax=ax, ls='-.', lw=0.5)
    plt_data['CloseBB5_-1.5'].plot(ax=ax, ls='--', lw=0.5)
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(), plt_data['Category'].values[np.newaxis], cmap='RdYlGn', alpha=0.3)
    # myFmt = dates.DateFormatter('%m-%d %H')
    # ax.xaxis.set_major_formatter(myFmt)
    # ax.xaxis.set_major_locator(dates.HourLocator(interval=20))
    ## Rotate date labels automatically
    fig.autofmt_xdate()
    matplotlib.rcParams.update({'font.size': 5})
    ax.legend(fontsize=4)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    #plt.ion()
    plt.show()


if __name__=='__main__':
    ticker = "BTC"
    new_data = False

    row_data = get_data(ticker, new_data)
    df = row_data.drop(['Open', 'High', 'Low'], axis=1)
    df = aggregate_indicator(df, column='Close', indicator='bollinger', length=10, num_stds=(1.5, 0, -1.5),
                             prefix='Close_BB5_', signal=True)
    df = aggregate_indicator(df, column='Close', indicator='delta_avg', length=10, prefix='delta_avg')
    df['std_10'] = df['Close'].rolling(10).std()
    df = df.iloc[10:]
    df = cat_to_dummies(df,'Signal', 'Market')
    df = df.drop(['Close_BB5_1.5', 'Close_BB5_0', 'Close_BB5_-1.5'], axis=1)
    cluster_cols = df.columns.to_list()
    cluster_cols.remove('Close')
    test_data = aggregate_categories(df, cluster_cols, n_clusters=25)
    print(test_data.head())


    #data['SMA5'] = data['Close'].rolling(window=5).mean()
    #data['SMA10'] = data['Close'].rolling(window=10).mean()
    #data['SMA20'] = data['Close'].rolling(window=20).mean()
    #data = data.iloc[20:]