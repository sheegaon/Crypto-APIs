"""
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

        return pd.DataFrame(dct_response.popitem()[1]['products'][0]['assets'])

    def get_token_balances(self, address, network='ethereum'):
        return self.get_balances(address=address, network=network, protocol='tokens')

    def get_uniswap_balances(self, address, network='ethereum'):
        return self.get_balances(address=address, network=network, protocol='uniswap-v3')

    def get_all_tokens(self, address):
        df_uni = self.get_uniswap_balances(address)
        df_wal = self.get_token_balances(address)
        cols = ['symbol', 'price', 'balance', 'balanceUSD']
        df_all = df_wal[cols].copy()
        df_all = pd.concat([df_all, df_uni.loc[df_uni['type'] == 'claimable', cols]])
        for _, tk in df_uni.loc[df_uni['type'] == 'pool', ['tokens']].iterrows():
            tkr = pd.DataFrame(tk['tokens'])
            df_all = pd.concat([df_all, tkr[cols]])
        df_sum = df_all.groupby('symbol')[['balance', 'balanceUSD']].sum()
        df_sum['price'] = df_all.drop_duplicates('symbol').set_index('symbol')['price']
        return df_sum
