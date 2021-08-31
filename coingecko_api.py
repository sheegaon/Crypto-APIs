import logging
import pandas as pd
from pycoingecko import CoinGeckoAPI
from datetime import datetime

STABLECOINS = ['DAI', 'USDC', 'FEI', 'USDT', 'UST', 'TUSD']
LST_DUPLICATE_COINS = ['unicorn-token', 'universe-token', 'millimeter', 'mm-token', 'compound-coin', 'raicoin',
                       'rai-finance', 'rai-token', 'truefi', 'game', 'gastrocoin', 'global-trust-coin', 'rune',
                       'thorchain-erc20', 'payperex', 'earnscoin', 'binance-peg-dogecoin', 'smartmesh',
                       'binance-peg-iotex', 'porkswap', 'euro-ritva-token', 'golden-ratio-token', 'hydro-protocol',
                       'hotnow', 'wrapped-terra', 'banana', 'banana-finance', 'supercoin', 'san-diego-coin',
                       'covenants', 'unifi-defi', 'wild-credit', 'wild-crypto', 'bitdao', 'bitrewards-token',
                       'first-bitcoin', 'bat-finance', 'arc-governance-old', 'oracolxor', 'rare', 'unique-one',
                       'solar-dao']
REFRESH_INTERVAL = 60 * 10  # Number of seconds until stale price data is re-downloaded
COINS = dict()


class CoinGecko:
    def __init__(self):
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
        symbols = list(set(symbols))
        symbols = [s.replace('wdoge', 'doge') for s in symbols]
        symbols = [s.replace('ewtb', 'ewt') for s in symbols]
        lst_valid = [s in self.df_coins['symbol'].to_list() for s in symbols]
        if not all(lst_valid):
            raise ValueError(f"Symbol {symbols[lst_valid.index(False)]} not found!")

        if f"{','.join(symbols)}_{vs_currencies}" in self.cache.keys():
            date, df_prc = self.cache[f"{','.join(symbols)}_{vs_currencies}"]
            if (datetime.today() - date).total_seconds() < REFRESH_INTERVAL:
                return df_prc

        ids = self.df_coins.set_index('symbol').loc[symbols, 'id'].to_list()
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
