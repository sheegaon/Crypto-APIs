import pandas as pd
import requests
import json
import os
from time import sleep
from datetime import datetime


def fetch_data(symbol, start='2019-01-01', granularity=3600):
    """
    Download historical data from Coinbase via API.

    Parameters
    ----------
    symbol : str
        Symbol as CCY1-CCY2.
    start : datetime or str
        Start date for downloading data.
    granularity : int, default 3600 (1h)
        Granularity in seconds.

    Returns
    -------
    pd.DataFrame
        DF with historical data.
    """
    start = pd.to_datetime(start)
    end = start + pd.Timedelta(12, 'd')
    all_data = []
    while end < datetime.now() + pd.Timedelta(12, 'd'):
        url = (f'https://api.pro.coinbase.com/products/{symbol}/candles?granularity={granularity}'
               f'&start={start:%Y-%m-%d}&end={end:%Y-%m-%d}')
        response = requests.get(url)
        if response.status_code == 404:
            start = end
            end = start + pd.Timedelta(12, 'd')
            sleep(.1)
            continue
        if response.status_code == 200:  # check to make sure the response from server is good
            df_dat = pd.DataFrame(json.loads(response.text), columns=['unix', 'low', 'high', 'open', 'close', 'volume'])
            # convert to a readable date
            df_dat['date'] = pd.to_datetime(df_dat['unix'], unit='s')

            # if we failed to get any data, print an error...otherwise append DF
            if df_dat is None:
                print('Did not return any data from Coinbase for this symbol')
            else:
                all_data += [df_dat]
            start = end
            end = start + pd.Timedelta(12, 'd')
            sleep(.1)
        else:
            print('Did not receieve OK response from Coinbase API')
            raise ValueError
    return pd.concat(all_data).drop_duplicates().sort_values('date')


def download_multiple_pairs(granularity=3600):
    import shared.misc_tools.misc_utils as mu

    # Download data for several commonly used pairs
    lst_symbols = [
        'BTC-USD', 'ETH-USD', 'LINK-USD', 'UNI-USD', 'AAVE-USD', 'MKR-USD', 'COMP-USD', 'MATIC-USD', 'CRV-USD',
        'SNX-USD', 'YFI-USD', '1INCH-USD', 'LINK-ETH', 'ETH-BTC']
    for symbol in lst_symbols:
        df_coin = fetch_data(symbol=symbol, granularity=granularity)
        folder, _ = os.path.split(__file__)
        folder = folder[:-4]
        mu.df_to_files(df_coin, os.path.join(folder, 'data', f'coinbase_{symbol.lower()}_{granularity}s'),
                       to_csv=False, to_pkl=False)


if __name__ == '__main__':
    download_multiple_pairs()
