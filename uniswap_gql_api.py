"""
  Utilities for accessing Uniswap's GraphQL subgraph at thegraph.com.
  Uniswap v3 liquidity math notes: https://medium.com/auditless/impermanent-loss-in-uniswap-v3-6c7161d3b445
"""
from datetime import datetime
import pandas as pd
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

FEE_TIER_TO_TICK_SPACING = {500: 10, 3000: 60, 10000: 200}
CACHE = dict()
CACHE_NOPKL = dict()
REDOWNLOAD_INTERVAL = 3600  # Number of seconds until stale date is re-downloaded
SAVE_CACHE_INTERVAL = 86400  # Number of seconds until cache is saved locally


def main():
    df_top = top_pools()
    pool_id = df_top.loc[0, 'id']
    plot_liquidity(pool_id)


# noinspection PyUnresolvedReferences
def get_pool_id(token0, token1, fee_tier):
    """
    Find the pool ID for a given token pair and fee tier.

    Parameters
    ----------
    token0 : str
        Ticker symbol of token 0.
    token1 : str
        Ticker symbol of token 1.
    fee_tier : int
        Fee tier in 100 x bp.

    Returns
    -------
    str
        Pool ID.
    """
    if 'pools' not in CACHE.keys():
        query = """
          query
            {{
              pools(first:1000, skip:{skip}, where: {{totalValueLockedUSD_gt: 1000, volumeUSD_gt: 100}}) {{
                id
                token0 {{
                  symbol
                }}
                token1 {{
                  symbol
                }}
                feeTier
              }}
            }}
            """
        df_pools = fetch_query_data(query)['pools']
        CACHE['pools'] = df_pools[['id', 'feeTier', 'token0.symbol', 'token1.symbol']].set_index('id')
    df_pools = CACHE['pools']
    ix = (df_pools['feeTier'] == fee_tier) & (df_pools['token0.symbol'] == token0.upper()) & (
            df_pools['token1.symbol'] == token1.upper())
    if not ix.any():
        ix = (df_pools['feeTier'] == fee_tier) & (df_pools['token0.symbol'] == token1.upper()) & (
                df_pools['token1.symbol'] == token0.upper())
    if ix.any():
        return df_pools[ix].index[0]
    raise ValueError(f"Pool with fee tier {fee_tier} and tokens {token0} and {token1} not found.")


def top_pools(num=500, order_by='totalValueLockedUSD'):
    if f'top_{num}_pools' in CACHE.keys():
        date, df_top = CACHE[f'top_{num}_pools']
        if (datetime.today() - date).total_seconds() < REDOWNLOAD_INTERVAL:
            return df_top

    query = f"""
        query
        {{
          pools(first:{num}, orderBy: {order_by}, orderDirection: desc, 
                where: {{totalValueLockedUSD_gt: 100000, volumeUSD_gt: 1000, txCount_gt: 500}}) {{
            id
            feeTier
            volumeUSD
            feesUSD
            txCount
            totalValueLockedUSD
            totalValueLockedETH
            token0 {{
              symbol
            }}
            token1 {{
              symbol
            }}
          }}
        }}
        """
    df_top = fetch_query_data(query)['pools']
    CACHE[f'top_{num}_pools'] = (datetime.today(), df_top)
    save_cache()
    return df_top


def fetch_pool_attributes(pool_id, latest=False):
    if f'att_{pool_id}' in CACHE.keys() and not latest:
        date, s_att = CACHE[f'att_{pool_id}']
        if (datetime.today() - date).total_seconds() < REDOWNLOAD_INTERVAL:
            return s_att

    from uniswap_tools import tick_to_price

    query = f"""
        query
        {{
          pools(where:{{id:"{pool_id}"}}) {{
            token0 {{
              symbol
              name
              decimals
            }}
            token1 {{
              symbol
              name
              decimals
            }}
            createdAtTimestamp
            feeTier
            liquidity
            sqrtPrice
            tick
            token0Price
            token1Price
            observationIndex
            volumeToken0
            volumeToken1
            volumeUSD
            feesUSD
            txCount
            totalValueLockedToken0
            totalValueLockedToken1
            totalValueLockedUSD            
          }}
        }}        
    """
    s_att = fetch_query_data(query)['pools'].loc[0, :]
    for c in s_att.index:
        v = pd.to_numeric(s_att[c], errors='coerce')
        if pd.notna(v):
            s_att[c] = v
    s_att['tick_spacing'] = FEE_TIER_TO_TICK_SPACING[s_att.loc['feeTier']]
    s_att['cur_prc'] = tick_to_price(
        s_att['tick'], decimals0=s_att['token0.decimals'], decimals1=s_att['token1.decimals'])
    CACHE[f'att_{pool_id}'] = (datetime.today(), s_att)
    save_cache()
    return s_att


def fetch_pool_liquidity(pool_id, fill_near=None, latest=False):
    if f'liq_{pool_id}' in CACHE.keys() and not latest:
        date, df_liq = CACHE[f'liq_{pool_id}']
        if (datetime.today() - date).total_seconds() < REDOWNLOAD_INTERVAL:
            return df_liq

    from uniswap_tools import tick_to_price, price_to_tick

    query = f"""
        query
        {{
          pools(where:{{id:"{pool_id}"}}) {{
            ticks(first:1000, skip:{{skip}}) {{
              liquidityGross
              liquidityNet
              tickIdx
              feeGrowthOutside0X128
              feeGrowthOutside1X128
            }}
            token0 {{
              symbol
              name
              decimals
            }}
            token1 {{
              symbol
              name
              decimals
            }}            
          }}
        }}
        """
    query = query.replace('{', '{{').replace('}', '}}').replace('{{skip}}', '{skip}')
    df_fetch = fetch_query_data(query)['pools']
    dec0 = pd.to_numeric(df_fetch['token0.decimals'].iloc[0])
    dec1 = pd.to_numeric(df_fetch['token1.decimals'].iloc[0])
    df_liq = df_fetch.iloc[0, 0].sort_values('tickIdx').set_index('tickIdx')
    if fill_near is not None:
        import numpy as np
        center_tick = price_to_tick(fill_near, decimals0=dec0, decimals1=dec1)
        ticks = np.array(df_liq.index)
        center_tick = ticks[np.argmin(np.abs(ticks - center_tick))]
        tick_interval = np.min(ticks[1:] - ticks[:-1])
        ticks = np.concatenate(
            [ticks,
             np.array(np.arange(center_tick - 40 * tick_interval, center_tick + 41 * tick_interval, tick_interval))])
        ticks = list(set(ticks))
        ticks.sort()
        df_liq = df_liq.reindex(ticks).fillna(0.)
    df_liq['feeGrowthOutside0X128'] = pd.to_numeric(df_liq['feeGrowthOutside0X128'], errors='coerce')
    df_liq['feeGrowthOutside1X128'] = pd.to_numeric(df_liq['feeGrowthOutside1X128'], errors='coerce')
    df_liq['prc0'] = tick_to_price(df_liq.index, decimals0=dec0, decimals1=dec1)
    df_liq['prc1'] = 1. / df_liq['prc0']
    df_liq['liquidityNet'] = pd.to_numeric(df_liq['liquidityNet'].apply(int) / 10 ** ((dec0 + dec1) / 2))
    df_liq['liquidity_total'] = df_liq['liquidityNet'].cumsum()
    df_liq['tick_width'] = df_liq['prc0'].shift(-1) - df_liq['prc0']
    CACHE[f'liq_{pool_id}'] = (datetime.today(), df_liq)
    save_cache()
    return df_liq


def fetch_pool_daily_data(pool_id, num_days=30):
    if f'daily_{pool_id}' in CACHE.keys():
        date, df_pool_history = CACHE[f'daily_{pool_id}']
        if (datetime.today() - date).total_seconds() < REDOWNLOAD_INTERVAL and num_days == 30:
            return df_pool_history
    else:
        df_pool_history = pd.DataFrame()

    start_dt = _unixtime(datetime.today().date() + pd.Timedelta(-num_days, 'd'))
    query = f"""
        query
        {{
          pools(where:{{id:"{pool_id}"}}) {{
            poolDayData(where:{{date_gt:{int(start_dt)}}}) {{
              date
              tvlUSD
              feesUSD
              volumeToken0
              volumeToken1
              volumeUSD
              txCount
              open
              high
              low
              close
            }}
          }}
        }}        
    """
    df_pool_history = pd.concat([df_pool_history, fetch_query_data(query)['pools'].iloc[0, 0]]).drop_duplicates(
        subset=['date'], keep='last').reset_index(drop=True)
    CACHE[f'daily_{pool_id}'] = (datetime.today(), df_pool_history)
    save_cache()
    return df_pool_history


def fetch_pool_hourly_data(pool_id, num_days=30):
    if f'hourly_{pool_id}' in CACHE.keys():
        date, df_pool_history = CACHE[f'hourly_{pool_id}']
        if (datetime.today() - date).total_seconds() < REDOWNLOAD_INTERVAL and num_days == 30:
            return df_pool_history
    else:
        df_pool_history = pd.DataFrame()

    start_dt = _unixtime(datetime.today().date() + pd.Timedelta(-num_days, 'd'))
    query = f"""
        query
        {{
          pools(where:{{id:"{pool_id}"}}) {{
            poolHourData(first:1000, skip:{{skip}}, where:{{periodStartUnix_gt:{int(start_dt)}}}) {{
              periodStartUnix
              tvlUSD
              txCount
              open
              high
              low
              close
            }}
          }}
        }}        
    """
    query = query.replace('{', '{{').replace('}', '}}').replace('{{skip}}', '{skip}')
    df_pool_history = pd.concat([df_pool_history, fetch_query_data(query)['pools'].iloc[0, 0]]).drop_duplicates(
        subset=['date'], keep='last').reset_index(drop=True)
    CACHE[f'hourly_{pool_id}'] = (datetime.today(), df_pool_history)
    save_cache()
    return df_pool_history


def fetch_mints_burns(origin=None, pool=None, mints=True, latest=False):
    qtype = 'mints' if mints else 'burns'
    if f'{qtype}_{origin}_{pool}' in CACHE.keys() and not latest:
        date, df_out = CACHE[f'{qtype}_{origin}_{pool}']
        if (datetime.today() - date).total_seconds() < REDOWNLOAD_INTERVAL:
            return df_out
    else:
        df_out = pd.DataFrame()

    where = []
    if origin is not None:
        where += [f'origin: "{origin}"']
    if pool is not None:
        where += [f'pool: "{pool}"']
    where = ', '.join(where)
    query = f"""
    query
    {{
      {qtype}(first: 1000, skip:{{skip}}, where: {{{where}}}) {{
        id
        transaction {{
          gasUsed
          gasPrice
        }}
        timestamp
        pool {{
          id
          feeTier
        }}
        token0 {{
          symbol
          decimals
        }}
        token1 {{
          symbol
          decimals
        }}
        owner
        origin
        amount
        amount0
        amount1
        amountUSD
        tickLower
        tickUpper
        logIndex
      }}
    }}
    """
    query = query.replace('{', '{{').replace('}', '}}').replace('{{skip}}', '{skip}')
    df_out = pd.concat([df_out, fetch_query_data(query)[qtype]]).drop_duplicates(
        subset=['id'], keep='last').reset_index(drop=True)
    CACHE[f'{qtype}_{origin}_{pool}'] = (datetime.today(), df_out)
    save_cache()
    return df_out


def get_active_positions(wallet_id):
    """
    Fetch all mints and burns for a given wallet and net out to find active positions.

    Parameters
    ----------
    wallet_id : str

    Returns
    -------
    pd.DataFrame
    """
    from uniswap_tools import tick_to_price, UniswapPosition

    df_mym = fetch_mints_burns(wallet_id, latest=True)
    df_mym['pos_id'] = df_mym['pool.id'] + df_mym['tickLower'].apply(str) + df_mym['tickUpper'].apply(str)
    df_active = df_mym.drop_duplicates(subset=['pos_id']).set_index('pos_id')
    df_mym['token0.decimals'] = pd.to_numeric(df_mym['token0.decimals'])
    df_mym['token1.decimals'] = pd.to_numeric(df_mym['token1.decimals'])
    for i, my in df_mym.iterrows():
        if my['amount'] == '0':
            continue
        pos = UniswapPosition(
            prc_lower=tick_to_price(my['tickLower'], decimals0=my['token0.decimals'], decimals1=my['token1.decimals']),
            prc_upper=tick_to_price(my['tickUpper'], decimals0=my['token0.decimals'], decimals1=my['token1.decimals']),
            x0=my['amount0'], y0=my['amount1'])
        df_mym.loc[i, 'liq'] = pos.liq
    df_active['mint_liq'] = df_mym.groupby('pos_id')['liq'].sum()
    df_myb = fetch_mints_burns(wallet_id, mints=False, latest=True)
    df_myb['pos_id'] = df_myb['pool.id'] + df_myb['tickLower'].apply(str) + df_myb['tickUpper'].apply(str)
    df_myb['token0.decimals'] = pd.to_numeric(df_myb['token0.decimals'])
    df_myb['token1.decimals'] = pd.to_numeric(df_myb['token1.decimals'])
    for i, my in df_myb.iterrows():
        if my['amount'] == '0':
            continue
        pos = UniswapPosition(
            prc_lower=tick_to_price(my['tickLower'], decimals0=my['token0.decimals'], decimals1=my['token1.decimals']),
            prc_upper=tick_to_price(my['tickUpper'], decimals0=my['token0.decimals'], decimals1=my['token1.decimals']),
            x0=my['amount0'], y0=my['amount1'])
        df_myb.loc[i, 'liq'] = pos.liq
    df_active['burn_liq'] = df_myb.groupby('pos_id')['liq'].sum()
    df_active['burn_liq'].fillna(0, inplace=True)
    df_active['remain_liq'] = 1 - df_active['burn_liq'] / df_active['mint_liq']
    df_active['amountUSD'] *= df_active['remain_liq']
    df_active = df_active[df_active['amountUSD'] > 1].copy()
    return df_active


def plot_liquidity(pool_id, center=None, range_factor=2, plot_type='stairs'):
    s_att = fetch_pool_attributes(pool_id)
    df_liq = fetch_pool_liquidity(pool_id)
    cur_prc = s_att['cur_prc']
    prc = 'prc0'
    if cur_prc < 1:
        prc = 'prc1'
        cur_prc = 1 / cur_prc
    if center is None:
        center = cur_prc
    lim = (df_liq[prc] > center / range_factor) & (df_liq[prc] < center * range_factor)

    if plot_type == 'line':
        df_liq[lim].plot.line(x=prc, y='liquidity_total')
    elif plot_type == 'stairs':
        import matplotlib.pyplot as plt
        plt.stairs(df_liq.loc[lim, 'liquidity_total'].values[:-1], df_liq.loc[lim, prc].values, fill=True)


def fetch_query_data(query):
    import numpy as np

    def result_to_df(lst_result):
        df_res = pd.DataFrame(lst_result)

        if 'date' in df_res.columns:
            df_res['unix'] = df_res['date']
            df_res['date'] = pd.to_datetime(df_res['unix'], unit='s')
            df_res.sort_values('date', inplace=True)

        for tvar in ['timestamp', 'periodStartUnix', 'createdAtTimestamp']:
            if tvar in df_res.columns:
                df_res['date'] = pd.to_datetime(df_res[tvar], unit='s')
                df_res.sort_values(tvar, inplace=True)

        for v in df_res.columns:
            test_v = pd.to_numeric(df_res[v].iloc[0], errors='ignore')

            if type(df_res[v].iloc[0]) is dict:
                for k in df_res[v].iloc[0].keys():
                    df_res[f"{v}.{k}"] = df_res[v].apply(lambda x: x[k])

            elif type(df_res[v].iloc[0]) is list:
                df_res[v] = df_res[v].apply(lambda x: result_to_df(x))

            elif type(test_v) == np.float64 or type(test_v) == np.int64:
                df_res[v] = pd.to_numeric(df_res[v], errors='coerce')
        return df_res

    if '{skip}' in query:
        skip = 0
        lst_res = []
        rk = ''
        res = {rk: 1000 * ['']}
        while len(res[rk]) == 1000:
            res = execute(query.format(skip=skip))
            rk = list(res.keys())[0]
            lst_res += res[rk]
            skip += 1000
        result = {rk: lst_res}
    else:
        result = execute(query)
    dct_result = dict()
    for schema in result.keys():
        df_schema = result_to_df(result[schema])
        dct_result[schema] = df_schema.reset_index(drop=True)

    return dct_result


def execute(query):
    try:
        client = _get_client()
        result = client.execute(gql(query))
    except:
        from time import sleep
        sleep(5)
        save_cache(force=True)
        _get_client(renew=True)
        return execute(query)
    return result


def save_cache(force=False):
    import os
    import pickle

    folder, _ = os.path.split(__file__)
    folder = folder[:-4]
    fn = os.path.join(folder, 'data', 'uniswap_gql_cache.pickle')
    if force:
        with open(fn, 'wb') as pkl_handle:
            pickle.dump(CACHE, pkl_handle)
        return

    from time import ctime
    fmod_time = pd.to_datetime(ctime(os.stat(fn).st_mtime))
    if os.path.isfile(fn) and (datetime.today() - fmod_time).total_seconds() > SAVE_CACHE_INTERVAL:
        with open(fn, 'wb') as pkl_handle:
            pickle.dump(CACHE, pkl_handle)


def load_cache():
    import os
    import pickle

    folder, _ = os.path.split(__file__)
    folder = folder[:-4]
    print(folder)
    fn = os.path.join(folder, 'data', 'uniswap_gql_cache.pickle')
    if os.path.isfile(fn):
        with open(fn, 'rb') as pkl_handle:
            cache = pickle.load(pkl_handle)
        for k in cache.keys():
            CACHE[k] = cache[k]


def _unixtime(date):
    return (pd.to_datetime(date) - datetime(1970, 1, 1)).total_seconds()


def _get_client(renew=False):
    if 'client' not in CACHE_NOPKL.keys() or renew:
        transport = RequestsHTTPTransport(
            url='https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
            verify=True,
            retries=5,
        )

        # Create a GraphQL client using the defined transport
        client = Client(transport=transport, fetch_schema_from_transport=True)
        CACHE_NOPKL['client'] = client
        load_cache()
    return CACHE_NOPKL['client']


if __name__ == '__main__':
    main()
