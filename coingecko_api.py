import os
import pickle
import logging
import pandas as pd
from pycoingecko import CoinGeckoAPI
from datetime import datetime

STABLECOINS = ['DAI', 'USDC', 'FEI', 'USDT', 'UST', 'TUSD', 'EURT', 'aUSDC', 'LUSD']
LST_DUPLICATE_COINS = [
    'unicorn-token', 'universe-token', 'millimeter', 'mm-token', 'compound-coin', 'raicoin', 'rai-finance', 'rai-token',
    'truefi', 'game', 'gastrocoin', 'global-trust-coin', 'rune', 'thorchain-erc20', 'payperex', 'earnscoin',
    'binance-peg-dogecoin', 'smartmesh', 'binance-peg-iotex', 'porkswap', 'euro-ritva-token', 'golden-ratio-token',
    'hydro-protocol', 'hotnow', 'wrapped-terra', 'banana', 'banana-finance', 'supercoin', 'san-diego-coin',
    'unifi-defi', 'wild-credit', 'wild-crypto', 'wild-ride', 'bitmoney', 'bitrewards-token', 'first-bitcoin',
    'bat-finance', 'arc-governance-old', 'oracolxor', 'rare', 'unique-one', 'solar-dao', 'dogeswap-token-heco',
    'capital-finance', 'fitmin', 'xpose-protocol', 'polkafantasy', 'paper', 'apoyield', 'soul-swap', 'cryptosoul',
    'aeneas', 'donkey-token', 'donnie-finance', 'deonex-token', 'iron-finance', 'decentral-games-ice', 'aave-usdc-v1',
    'robinos', 'dogswap-token', 'genesis-mana', 'rinnegan', 'margin-call', 'impermax', 'floki-one', 'floki-musk',
    'shiba-floki', 'baby-moon-floki', 'staked-olympus', 'ash', 'aeneas', 'ufocoin', 'unknown-fair-object',
    'polybeta-finance', 'mim', 'koinon', 'heliumx', 'osmoscoin', 'metaaxis', 'wolf-game-wool', 'tower-finance',
    'safeswap-online', 'juicebox', 'cardswap', 'umi', 'decentral-games-ice', 'ice-dao', 'game-fantasy-token',
    'good-fire', 'olympus-v1',
]
REFRESH_INTERVAL = 60 * 10  # Number of seconds until stale price data is re-downloaded
COINS = dict()
CACHE_INTERVAL = 86400  # Number of seconds until history cache is saved to file


class CoinGecko:
    def __init__(self):
        # Load cached data
        folder, _ = os.path.split(__file__)
        folder = folder[:-4]
        self.fn = os.path.join(folder, 'data', 'coingecko_history_cache.pickle')
        if os.path.isfile(self.fn):
            try:
                with open(self.fn, 'rb') as pkl_handle:
                    self.history = pickle.load(pkl_handle)
            except pickle.UnpicklingError:
                self.history = dict()
        else:
            self.history = dict()
        self.last_saved = datetime.now()

        self.cg_api = CoinGeckoAPI()
        if 'df' in COINS.keys():
            self.df_coins = COINS['df']
        else:
            lst_coins = self.cg_api.get_coins_list()
            self.df_coins = pd.DataFrame(lst_coins)
            # Remove some duplicates
            for i in LST_DUPLICATE_COINS:
                self.df_coins = self.df_coins[self.df_coins['id'] != i].copy()
            COINS['df'] = self.df_coins
        self.cache = dict()

    def save_history(self, force=False):
        if force or (datetime.now() - self.last_saved).total_seconds() > CACHE_INTERVAL:
            with open(self.fn, 'wb') as pkl_handle:
                pickle.dump(self.history, pkl_handle)
            self.last_saved = datetime.now()

    def get_id(self, symbol):
        if isinstance(symbol, list):
            symbols = [s.lower() for s in symbol]
            symbols = list(set(symbols))
            symbols = [s.replace('wdoge', 'doge') for s in symbols]
            symbols = [s.replace('ewtb', 'ewt') for s in symbols]
            lst_valid = [s in self.df_coins['symbol'].to_list() for s in symbols]
            if not all(lst_valid):
                raise ValueError(f"Symbol {symbols[lst_valid.index(False)]} not found!")

            cg_ids = self.df_coins.set_index('symbol').loc[symbols, 'id'].to_list()
            return cg_ids

        symbol = symbol.lower()
        symbol = 'doge' if symbol == 'wdoge' else symbol
        symbol = 'ewt' if symbol == 'ewtb' else symbol
        if symbol not in self.df_coins['symbol'].to_list():
            raise ValueError(f"Symbol {symbol} not found!")

        cg_id = self.df_coins.set_index('symbol').loc[symbol, 'id']
        return cg_id

    def get_price(self, symbols, vs_currencies='usd', **kwargs):
        """
        Retrieve price via CoinGecko API

        Parameters
        ----------
        symbols : list
        vs_currencies : str
        kwargs

        Returns
        -------
        pd.DataFrame
        """
        ids = self.get_id(symbols)
        if f"{','.join(symbols)}_{vs_currencies}" in self.cache.keys():
            date, df_prc = self.cache[f"{','.join(symbols)}_{vs_currencies}"]
            if (datetime.today() - date).total_seconds() < REFRESH_INTERVAL:
                return df_prc
        try:
            df_prc = pd.DataFrame(self.cg_api.get_price(ids=ids, vs_currencies=vs_currencies, **kwargs))
        except:
            from time import sleep
            sleep(1)
            self.cg_api = CoinGeckoAPI()
            return self.get_price(symbols=symbols, vs_currencies=vs_currencies, **kwargs)
        df_prc.columns = self.df_coins.set_index('id').loc[df_prc.columns, 'symbol'].to_list()
        if df_prc.columns.duplicated().any():
            logging.warning(f"Duplicate symbols in CoinGecko API request for {symbols}.")
            df_prc = df_prc.loc[:, ~df_prc.columns.duplicated()]

        # Replace some non-Ethereum coins with Uniswap ERC-20 versions
        df_prc.columns = [s.replace('doge', 'wdoge') for s in df_prc.columns]
        df_prc.columns = [s.replace('ewt', 'ewtb') for s in df_prc.columns]

        self.cache[f"{','.join(symbols)}_{vs_currencies}"] = (datetime.today(), df_prc)
        return df_prc

    def get_history(self, symbol, vs_currency='usd', days=5, interval='hourly', from_dt=None, to_dt=None, **kwargs):
        """
        Retrieve price via CoinGecko API

        Parameters
        ----------
        symbol : str
        vs_currency : str
        days : int
        interval : {'daily', 'hourly'}
        from_dt : datetime or str
        to_dt : datetime or str
        kwargs

        Returns
        -------
        pd.DataFrame
        """
        if (symbol.upper(), vs_currency.upper()) in self.history.keys():
            df_history = self.history[(symbol.upper(), vs_currency.upper())].copy()
        else:
            df_history = pd.DataFrame()

        if df_history.empty:
            is_contained = False
        elif from_dt is not None and to_dt is not None:
            is_contained = (df_history.index[0] - pd.to_datetime(from_dt) < pd.Timedelta(1, 'h')) and \
                           (df_history.index[-1] - pd.to_datetime(to_dt) <  pd.Timedelta(1, 'h'))
        else:
            if datetime.now() - df_history.index[-1] > pd.Timedelta(1, 'h'):
                is_contained = False
            else:
                is_contained = datetime.now() - pd.Timedelta(days*.99, 'd') > df_history.index[0]
        if not is_contained:
            cg_id = self.get_id(symbol)
            try:
                if from_dt is not None and to_dt is not None:
                    from_ts = timestamp(from_dt)
                    to_ts = timestamp(to_dt)
                    res = self.cg_api.get_coin_market_chart_range_by_id(
                        id=cg_id, vs_currency=vs_currency, from_timestamp=from_ts, to_timestamp=to_ts,
                        interval=interval, **kwargs)
                else:
                    res = self.cg_api.get_coin_market_chart_by_id(
                        id=cg_id, vs_currency=vs_currency, days=days, interval=interval, **kwargs)
                df_new_history = pd.DataFrame()
                for key, val in res.items():
                    df_new_history[key[:-1]] = pd.DataFrame(val, columns=['timestamp', key]).set_index('timestamp')[key]
                df_new_history.index = pd.to_datetime(df_new_history.index.to_list(), unit='ms')
            except Exception as ex:
                logging.warning(ex)
                from time import sleep
                sleep(5)
                self.cg_api = CoinGeckoAPI()
                return self.get_history(
                    symbol=symbol, vs_currency=vs_currency, days=days, interval=interval, from_dt=from_dt, to_dt=to_dt,
                    **kwargs)

            df_history = pd.concat([df_history, df_new_history]).drop_duplicates().sort_index()
            self.history[(symbol.upper(), vs_currency.upper())] = df_history.copy()
            self.save_history()

        return df_history

    def get_contract_address(self, symbol):
        return self.cg_api.get_coin_by_id(id=self.get_id(symbol))['contract_address']


def timestamp(dt):
    return f"{(pd.to_datetime(dt) - datetime(1970, 1, 1)).total_seconds():.0f}"
