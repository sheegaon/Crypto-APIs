"""
 This API is a python wrapper around Zapper's REST API, focusing on the functions I use most.

 https://docs.zapper.fi/zapper-api/endpoints
 https://api.zapper.fi/api/static/index.html
"""
import json
import requests
import logging

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import pandas as pd


class ZapperAPI:
    __API_URL_BASE = 'https://api.zapper.fi/v1/'
    __PUBLIC_KEY = '96e0cc51-a62e-42ca-acee-910ea7d2a241'

    def __init__(self, api_base_url=__API_URL_BASE, api_key=__PUBLIC_KEY):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.request_timeout = 120

        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def __request(self, url):
        try:
            response = self.session.get(url, timeout=self.request_timeout)
        except requests.exceptions.RequestException:
            raise

        try:
            response.raise_for_status()
            content = json.loads(response.content.decode('utf-8'))
            return content
        except Exception as e:
            # check if json (with error message) is returned
            try:
                from time import sleep
                content = json.loads(response.content.decode('utf-8'))
                if content['message'] == 'Request Timeout':
                    logging.warning(content)
                    sleep(5)
                    return self.__request(url)
                else:
                    raise ValueError(content)
            # if no json
            except json.decoder.JSONDecodeError:
                pass

            raise

    def __api_url_params(self, api_url, params, api_url_has_params=False):
        if params:
            # if api_url contains already params and there is already a '?' avoid
            # adding second '?' (api_url += '&' if '?' in api_url else '?'); causes
            # issues with request parametes (usually for endpoints with required
            # arguments passed as parameters)
            api_url += '&' if api_url_has_params else '?'
            for key, value in params.items():
                if type(value) == bool:
                    value = str(value).lower()

                api_url += f"{key}={value}&"
            api_url = api_url[:-1]
        return api_url

    def get_balances(self, address, protocol, network='ethereum'):
        api_url = f'{self.api_base_url}protocols/{protocol}/balances'
        api_url = self.__api_url_params(
            api_url, {'api_key': self.api_key, 'addresses%5B%5D': address, 'network': network})

        dct_response = self.__request(api_url)

        try:
            return pd.DataFrame(dct_response.popitem()[1]['products'][0]['assets'])
        except:
            from time import sleep
            sleep(5)
            return self.get_balances(address=address, protocol=protocol, network=network)

    def get_token_balances(self, address, network='ethereum'):
        df_bal = self.get_balances(address=address, network=network, protocol='tokens')
        df_bal['symbol'] = df_bal['symbol'].str.replace('(v2)', '', regex=False)
        df_bal['category'] = 'wallet'
        df_bal['hide'].fillna(False, inplace=True)
        return df_bal

    def get_protocol_balances(self, address, protocol, network='ethereum'):
        """
        Get balances at a given address for a given protocol and network. Unpacks claimable tokens into main DataFrame.

        Parameters
        ----------
        address : str
            Ethereum wallet address.
        protocol : {'uniswap-v3', 'aave-v2'}
            More protocols may be supported over time as needed.
        network : {'ethereum', 'optimism', 'arbitrum', 'polygon'}
            Only ethereum network is guaranteed to work.

        Returns
        -------
        pd.DataFrame
        """
        df_bal = self.get_balances(address=address, network=network, protocol=protocol)
        for i, r in df_bal.iterrows():
            if r['type'] != 'claimable':
                continue
            for key, val in r['tokens'][0].items():
                if key == 'type':
                    continue
                df_bal.loc[i, key] = val
        return df_bal

    def get_all_tokens(self, address, exposures=False):
        df_uni = self.get_protocol_balances(address, protocol='uniswap-v3')
        df_aave = self.get_protocol_balances(address, protocol='aave-v2')
        df_wal = self.get_token_balances(address)
        cols = ['category', 'symbol', 'price', 'balance', 'balanceUSD']
        df_all = df_wal.loc[~df_wal['hide'], cols].copy()
        df_all = pd.concat([df_all, df_aave[cols], df_uni.loc[df_uni['category'] == 'claimable', cols]])
        df_all_exp = df_all.copy()
        for _, tk in df_uni.loc[df_uni['category'] == 'pool', ['tokens']].iterrows():
            df_tkr = pd.DataFrame(tk['tokens'])
            df_tkr['category'] = 'pool'
            df_all = pd.concat([df_all, df_tkr[cols]])
            if exposures:
                lst_sym1 = [x for x in df_tkr['symbol'] if x not in ['ETH', 'USDC', 'WETH', 'USDT']]
                if len(lst_sym1) == 0:
                    df_tkr['symbol'] = 'ETH'
                else:
                    df_tkr['symbol'] = [x for x in df_tkr['symbol'] if x not in ['ETH', 'USDC', 'WETH', 'USDT']][0]
                df_tkr['balanceUSD'] = df_tkr['balanceUSD'].sum()
                df_all_exp = pd.concat([df_all_exp, df_tkr.loc[[0], cols]])
        df_sum = df_all.groupby(['symbol', 'category'])[['balance', 'balanceUSD']].sum().reset_index().set_index(
            'symbol')
        df_sum['price'] = df_all.drop_duplicates('symbol').set_index('symbol')['price']
        if exposures:
            from api.coingecko_api import STABLECOINS

            df_all_exp['symbol'] = df_all_exp['symbol'].str.replace('a', '')
            s_pot_exp = df_all_exp.groupby('symbol')['balanceUSD'].sum()
            s_pot_exp['ETH'] += s_pot_exp['WETH']
            s_pot_exp.drop('WETH', inplace=True)
            s_pot_exp.drop(STABLECOINS, errors='ignore', inplace=True)
            s_pot_exp.sort_values(ascending=False, inplace=True)
            df_cur_exp = df_sum.reset_index()[['symbol', 'balanceUSD']].copy()
            df_cur_exp['symbol'] = df_cur_exp['symbol'].str.replace('WETH', 'ETH')
            df_cur_exp['symbol'] = df_cur_exp['symbol'].str.replace('a', '')
            s_cur_exp = df_cur_exp.groupby('symbol')['balanceUSD'].sum().sort_values(ascending=False)
            s_cur_exp.drop(STABLECOINS, errors='ignore', inplace=True)
            s_cur_exp.index.name = 'symbol'
            return df_sum, s_pot_exp, s_cur_exp
        return df_sum
