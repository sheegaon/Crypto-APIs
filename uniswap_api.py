"""
  A class-based API for accessing Uniswap's GraphQL subgraph at thegraph.com.
  Uniswap v3 liquidity math notes: https://medium.com/auditless/impermanent-loss-in-uniswap-v3-6c7161d3b445
"""
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

FEE_TIER_TO_TICK_SPACING = {500: 10, 3000: 60, 10000: 200}
DOWNLOAD_INTERVAL = 3600  # Number of seconds until stale date is re-downloaded
CACHE_INTERVAL = 86400  # Number of seconds until GQL cache is saved to file


class UniswapPosition:
    def __init__(self,
                 prc_lower,
                 prc_upper,
                 x0=None,
                 y0=None,
                 prc_init=None,
                 x_symbol=None,
                 y_symbol=None,
                 usd_init=None,
                 df_usd_prc=None):
        """
        Initialize a Uniswap v3 position object.
        All prices are the prices of token x in units of token y. For example, for the USDC/WETH pool, prices
        are in the vicinity of 0.0004, whereas for the WBTC/WETH pool prices are in the vicinity of 15.

        Parameters
        ----------
        prc_lower : float
            Lower bound of liquidity range.
        prc_upper : float
            Upper bound of liquidity range.
        x0 : float, optional
            Quantity of token x invested.
        y0 : float, optional
            Quantity of token y invested.
        prc_init : float
            Initial price when liquidity position was established.
        x_symbol : str
            Ticker symbol of token x.
        y_symbol : str
            Ticker symbol of token y.
        usd_init
        df_usd_prc : pd.DataFrame
        """
        assert prc_lower < prc_upper, "prc_lower must be less than prc_upper!"
        if x_symbol is not None:
            self.x_symbol = x_symbol
        if y_symbol is not None:
            self.y_symbol = y_symbol
        liq = None
        if x0 is None or y0 is None:
            assert prc_init is not None, "Must specify prc_init if x0 or y0 is None!"
            if x0 is not None:
                # Derive y0
                if prc_init < prc_lower:
                    y0 = 0
                elif prc_init > prc_upper:
                    raise ValueError("x0 = 0 if prc_init > prc_upper!")
                else:
                    liq = x0 / (1 / np.sqrt(prc_init) - 1 / np.sqrt(prc_upper))
                    x_virt0 = x0 + liq / np.sqrt(prc_upper)
                    y_virt0 = liq ** 2 / x_virt0
                    y0 = y_virt0 - liq * np.sqrt(prc_lower)
            elif y0 is not None:
                # Derive x0
                if prc_init > prc_upper:
                    x0 = 0
                elif prc_init < prc_lower:
                    raise ValueError("y0 = 0 if prc_init < prc_lower!")
                else:
                    liq = y0 / (np.sqrt(prc_init) - np.sqrt(prc_lower))
                    y_virt0 = y0 + liq * np.sqrt(prc_lower)
                    x_virt0 = liq ** 2 / y_virt0
                    x0 = x_virt0 - liq / np.sqrt(prc_upper)
            elif usd_init is not None:
                px = df_usd_prc.loc['usd', x_symbol]
                py = df_usd_prc.loc['usd', y_symbol]
                if prc_init < prc_lower:
                    y0 = 0
                    x0 = usd_init / px
                elif prc_init > prc_upper:
                    x0 = 0
                    y0 = usd_init / py
                else:
                    x_coef = (1 - np.sqrt(prc_init / prc_upper))
                    y_coef = (1 - np.sqrt(prc_lower / prc_init))
                    x_virt0, y_virt0 = np.linalg.solve(
                        np.array([[-prc_init, 1], [x_coef * px, y_coef * py]]),
                        np.array([0, usd_init]))
                    liq = np.sqrt(x_virt0 * y_virt0)
                    x0 = x_virt0 * x_coef
                    y0 = y_virt0 * y_coef
            else:
                raise ValueError("Must specify x0 or y0!")
        self.x0 = x0
        self.y0 = y0
        self.prc_lower = prc_lower
        self.prc_upper = prc_upper
        if liq is None:
            a = 1 - np.sqrt(prc_lower / prc_upper)
            b = - y0 / np.sqrt(prc_upper) - x0 * np.sqrt(prc_lower)
            c = - x0 * y0
            liq = np.roots([a, b, c])[0]
        self.liq = liq
        self.x_virt0 = self.x0 + self.liq / np.sqrt(self.prc_upper)
        self.y_virt0 = self.y0 + self.liq * np.sqrt(self.prc_lower)
        self.prc_init = self.y_virt0 / self.x_virt0
        self.tick_lower = None
        self.tick_upper = None
        self.cg = None

    def max_x(self):
        return self.liq * (1 / np.sqrt(self.prc_lower) - 1 / np.sqrt(self.prc_upper))

    def max_y(self):
        return self.liq * (np.sqrt(self.prc_upper) - np.sqrt(self.prc_lower))

    def set_ticks(self, decimals0, decimals1):
        self.tick_lower = prc2tick(self.prc_lower, decimals0, decimals1)
        self.tick_upper = prc2tick(self.prc_upper, decimals0, decimals1)

    def calc_reserves(self, prc):
        if prc <= self.prc_lower:
            x = self.max_x()
            y = 0
        elif prc >= self.prc_upper:
            x = 0
            y = self.max_y()
        else:
            x = self.liq * (1 / np.sqrt(prc) - 1 / np.sqrt(self.prc_upper))
            y = self.liq * (np.sqrt(prc) - np.sqrt(self.prc_lower))
        return x, y

    def current_value(self, x_symbol=None, y_symbol=None, df_prc=None):
        if x_symbol is not None:
            self.x_symbol = x_symbol
        if y_symbol is not None:
            self.y_symbol = y_symbol
        assert self.x_symbol is not None, "Must specify x_symbol!"
        assert self.y_symbol is not None, "Must specify y_symbol!"

        if df_prc is None:
            if self.cg is None:
                from api.coingecko_api import CoinGecko
                self.cg = CoinGecko()
            df_prc = self.cg.get_price(symbols=[self.x_symbol, self.y_symbol], vs_currencies='usd')
        prc = df_prc.loc['usd', self.x_symbol] / df_prc.loc['usd', self.y_symbol]
        x, y = self.calc_reserves(prc)
        return x * df_prc.loc['usd', self.x_symbol] + y * df_prc.loc['usd', self.y_symbol]

    def impermanent_loss(self, x_symbol=None, y_symbol=None, relative_to='init', df_prc=None):
        if x_symbol is not None:
            self.x_symbol = x_symbol
        if y_symbol is not None:
            self.y_symbol = y_symbol
        assert self.x_symbol is not None, "Must specify x_symbol!"
        assert self.y_symbol is not None, "Must specify y_symbol!"

        if df_prc is None:
            if self.cg is None:
                from api.coingecko_api import CoinGecko
                self.cg = CoinGecko()
            df_prc = self.cg.get_price(symbols=[self.x_symbol, self.y_symbol], vs_currencies='usd')
        prc = df_prc.loc['usd', self.x_symbol] / df_prc.loc['usd', self.y_symbol]
        x, y = self.calc_reserves(prc)
        if relative_to == 'init':
            val_ref = self.x0 * df_prc.loc['usd', self.x_symbol] + self.y0 * df_prc.loc['usd', self.y_symbol]
        elif relative_to == 'y':
            val_ref = (self.x0 * self.prc_init + self.y0) * df_prc.loc['usd', self.y_symbol]
        elif relative_to == 'x':
            val_ref = (self.x0 + self.y0 / self.prc_init) * df_prc.loc['usd', self.x_symbol]
        else:
            raise ValueError(f"Invalid relative_to '{relative_to}'")
        val_pos = x * df_prc.loc['usd', self.x_symbol] + y * df_prc.loc['usd', self.y_symbol]
        return val_ref - val_pos


class UniswapGQL:
    def __init__(self):
        self.client = _get_client()
    
        # Load cached data
        folder, _ = os.path.split(__file__)
        folder = folder[:-4]
        self.fn = os.path.join(folder, 'data', 'uniswap_gql_cache.pickle')
        if os.path.isfile(self.fn):
            with open(self.fn, 'rb') as pkl_handle:
                self.cache = pickle.load(pkl_handle)
        else:
            self.cache = dict()
        self.last_saved = datetime.now()

    def save_cache(self, force=False):
        if force or (datetime.now() - self.last_saved).total_seconds() > CACHE_INTERVAL:
            with open(self.fn, 'wb') as pkl_handle:
                pickle.dump(self.cache, pkl_handle)
            self.last_saved = datetime.now()

    def execute(self, query):
        try:
            result = self.client.execute(gql(query))
        except:
            from time import sleep
            sleep(5)
            self.save_cache(force=True)
            self.client = _get_client()
            return self.execute(query)
        return result

    def fetch_query_data(self, query):
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
                res = self.execute(query.format(skip=skip))
                rk = list(res.keys())[0]
                lst_res += res[rk]
                skip += 1000
            result = {rk: lst_res}
        else:
            result = self.execute(query)
        dct_result = dict()
        for schema in result.keys():
            df_schema = result_to_df(result[schema])
            dct_result[schema] = df_schema.reset_index(drop=True)

        return dct_result

    # noinspection PyUnresolvedReferences
    def get_pool_id(self, token0, token1, fee_tier):
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
        if 'pools' not in self.cache.keys():
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
            df_pools = self.fetch_query_data(query)['pools']
            self.cache['pools'] = df_pools[['id', 'feeTier', 'token0.symbol', 'token1.symbol']].set_index('id')
        df_pools = self.cache['pools']
        ix = (df_pools['feeTier'] == fee_tier) & (df_pools['token0.symbol'] == token0.upper()) & (
                df_pools['token1.symbol'] == token1.upper())
        if not ix.any():
            ix = (df_pools['feeTier'] == fee_tier) & (df_pools['token0.symbol'] == token1.upper()) & (
                    df_pools['token1.symbol'] == token0.upper())
        if ix.any():
            return df_pools[ix].index[0]
        raise ValueError(f"Pool with fee tier {fee_tier} and tokens {token0} and {token1} not found.")

    def top_pools(self, num=500, order_by='totalValueLockedUSD'):
        if f'top_{num}_pools' in self.cache.keys():
            date, df_top = self.cache[f'top_{num}_pools']
            if (datetime.today() - date).total_seconds() < DOWNLOAD_INTERVAL:
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
        df_top = self.fetch_query_data(query)['pools']
        self.cache[f'top_{num}_pools'] = (datetime.today(), df_top)
        self.save_cache()
        return df_top

    def fetch_pool_attributes(self, pool_id, latest=False):
        if f'att_{pool_id}' in self.cache.keys() and not latest:
            date, s_att = self.cache[f'att_{pool_id}']
            if (datetime.today() - date).total_seconds() < DOWNLOAD_INTERVAL:
                return s_att

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
        s_att = self.fetch_query_data(query)['pools'].loc[0, :]
        for c in s_att.index:
            v = pd.to_numeric(s_att[c], errors='coerce')
            if pd.notna(v):
                s_att[c] = v
        s_att['tick_spacing'] = FEE_TIER_TO_TICK_SPACING[s_att.loc['feeTier']]
        s_att['cur_prc'] = tick2prc(
            s_att['tick'], decimals0=s_att['token0.decimals'], decimals1=s_att['token1.decimals'])
        self.cache[f'att_{pool_id}'] = (datetime.today(), s_att)
        self.save_cache()
        return s_att

    def fetch_pool_liquidity(self, pool_id, fill_near=None, latest=False):
        if f'liq_{pool_id}' in self.cache.keys() and not latest:
            date, df_liq = self.cache[f'liq_{pool_id}']
            if (datetime.today() - date).total_seconds() < DOWNLOAD_INTERVAL:
                return df_liq

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
        df_fetch = self.fetch_query_data(query)['pools']
        dec0 = pd.to_numeric(df_fetch['token0.decimals'].iloc[0])
        dec1 = pd.to_numeric(df_fetch['token1.decimals'].iloc[0])
        df_liq = df_fetch.iloc[0, 0].sort_values('tickIdx').set_index('tickIdx')
        if fill_near is not None:
            center_tick = prc2tick(fill_near, decimals0=dec0, decimals1=dec1)
            ticks = np.array(df_liq.index)
            center_tick = ticks[np.argmin(np.abs(ticks - center_tick))]
            tick_interval = np.min(ticks[1:] - ticks[:-1])
            ticks = np.concatenate(
                [ticks,
                 np.array(np.arange(center_tick - 40 * tick_interval,
                                    center_tick + 41 * tick_interval, tick_interval))])
            ticks = list(set(ticks))
            ticks.sort()
            df_liq = df_liq.reindex(ticks).fillna(0.)
        df_liq['feeGrowthOutside0X128'] = pd.to_numeric(df_liq['feeGrowthOutside0X128'], errors='coerce')
        df_liq['feeGrowthOutside1X128'] = pd.to_numeric(df_liq['feeGrowthOutside1X128'], errors='coerce')
        df_liq['prc0'] = tick2prc(df_liq.index, decimals0=dec0, decimals1=dec1)
        df_liq['prc1'] = 1. / df_liq['prc0']
        df_liq['liquidityNet'] = pd.to_numeric(df_liq['liquidityNet'].apply(int) / 10 ** ((dec0 + dec1) / 2))
        df_liq['liquidity_total'] = df_liq['liquidityNet'].cumsum()
        df_liq['tick_width'] = df_liq['prc0'].shift(-1) - df_liq['prc0']
        self.cache[f'liq_{pool_id}'] = (datetime.today(), df_liq)
        self.save_cache()
        return df_liq

    def fetch_pool_daily_data(self, pool_id, num_days=30):
        if f'daily_{pool_id}' in self.cache.keys():
            date, df_pool_history = self.cache[f'daily_{pool_id}']
            if (datetime.today() - date).total_seconds() < DOWNLOAD_INTERVAL and num_days == 30:
                return df_pool_history
        else:
            df_pool_history = pd.DataFrame()

        start_dt = _unix_time(datetime.today().date() + pd.Timedelta(-num_days, 'd'))
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
        df_pool_history = pd.concat([
            df_pool_history, self.fetch_query_data(query)['pools'].iloc[0, 0]]).drop_duplicates(
            subset=['date'], keep='last').reset_index(drop=True)
        self.cache[f'daily_{pool_id}'] = (datetime.today(), df_pool_history)
        self.save_cache()
        return df_pool_history

    def fetch_pool_hourly_data(self, pool_id, num_days=30):
        if f'hourly_{pool_id}' in self.cache.keys():
            date, df_pool_history = self.cache[f'hourly_{pool_id}']
            if (datetime.today() - date).total_seconds() < DOWNLOAD_INTERVAL and num_days == 30:
                return df_pool_history
        else:
            df_pool_history = pd.DataFrame()

        start_dt = _unix_time(datetime.today().date() + pd.Timedelta(-num_days, 'd'))
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
        df_pool_history = pd.concat(
            [df_pool_history, self.fetch_query_data(query)['pools'].iloc[0, 0]]).drop_duplicates(
            subset=['date'], keep='last').reset_index(drop=True)
        self.cache[f'hourly_{pool_id}'] = (datetime.today(), df_pool_history)
        self.save_cache()
        return df_pool_history

    def fetch_mints_burns(self, origin=None, pool=None, mints=True, latest=False):
        qtype = 'mints' if mints else 'burns'
        if f'{qtype}_{origin}_{pool}' in self.cache.keys() and not latest:
            date, df_out = self.cache[f'{qtype}_{origin}_{pool}']
            if (datetime.today() - date).total_seconds() < DOWNLOAD_INTERVAL:
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
        df_out = pd.concat([df_out, self.fetch_query_data(query)[qtype]]).drop_duplicates(
            subset=['id'], keep='last').reset_index(drop=True)
        self.cache[f'{qtype}_{origin}_{pool}'] = (datetime.today(), df_out)
        self.save_cache()
        return df_out

    def get_active_positions(self, wallet_id):
        """
        Fetch all mints and burns for a given wallet and net out to find active positions.

        Parameters
        ----------
        wallet_id : str

        Returns
        -------
        pd.DataFrame
        """
        df_mym = self.fetch_mints_burns(wallet_id, latest=True)
        df_mym['pos_id'] = df_mym['pool.id'] + df_mym['tickLower'].apply(str) + df_mym['tickUpper'].apply(str)
        df_active = df_mym.drop_duplicates(subset=['pos_id']).set_index('pos_id')
        df_mym['token0.decimals'] = pd.to_numeric(df_mym['token0.decimals'])
        df_mym['token1.decimals'] = pd.to_numeric(df_mym['token1.decimals'])
        for i, my in df_mym.iterrows():
            if my['amount'] == '0':
                continue
            pos = UniswapPosition(
                prc_lower=tick2prc(my['tickLower'], decimals0=my['token0.decimals'], decimals1=my['token1.decimals']),
                prc_upper=tick2prc(my['tickUpper'], decimals0=my['token0.decimals'], decimals1=my['token1.decimals']),
                x0=my['amount0'], y0=my['amount1'])
            df_mym.loc[i, 'liq'] = pos.liq
        df_active['mint_liq'] = df_mym.groupby('pos_id')['liq'].sum()
        df_myb = self.fetch_mints_burns(wallet_id, mints=False, latest=True)
        df_myb['pos_id'] = df_myb['pool.id'] + df_myb['tickLower'].apply(str) + df_myb['tickUpper'].apply(str)
        df_myb['token0.decimals'] = pd.to_numeric(df_myb['token0.decimals'])
        df_myb['token1.decimals'] = pd.to_numeric(df_myb['token1.decimals'])
        for i, my in df_myb.iterrows():
            if my['amount'] == '0':
                continue
            pos = UniswapPosition(
                prc_lower=tick2prc(my['tickLower'], decimals0=my['token0.decimals'], decimals1=my['token1.decimals']),
                prc_upper=tick2prc(my['tickUpper'], decimals0=my['token0.decimals'], decimals1=my['token1.decimals']),
                x0=my['amount0'], y0=my['amount1'])
            df_myb.loc[i, 'liq'] = pos.liq
        df_active['burn_liq'] = df_myb.groupby('pos_id')['liq'].sum()
        df_active['burn_liq'].fillna(0, inplace=True)
        df_active['remain_liq'] = 1 - df_active['burn_liq'] / df_active['mint_liq']
        df_active['amountUSD'] *= df_active['remain_liq']
        df_active = df_active[df_active['amountUSD'] > 1].copy()
        return df_active

    def plot_liquidity(self, pool_id, center=None, range_factor=2, plot_type='stairs'):
        s_att = self.fetch_pool_attributes(pool_id)
        df_liq = self.fetch_pool_liquidity(pool_id)
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


def _unix_time(date):
    return (pd.to_datetime(date) - datetime(1970, 1, 1)).total_seconds()


def _get_client():
    transport = RequestsHTTPTransport(
        url='https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
        verify=True,
        retries=5,
    )

    # Create a GraphQL client using the defined transport
    client = Client(transport=transport, fetch_schema_from_transport=True)
    return client


def tick2prc(tick, decimals0, decimals1):
    return 1.0001 ** tick * (10. ** (decimals0 - decimals1))


def prc2tick(price, decimals0, decimals1):
    return int(np.round(np.log(price * 10. ** (decimals1 - decimals0)) / np.log(1.0001)))
