"""
 https://0x.org/docs/api
"""
import json
import requests
import logging

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import pandas as pd


class ZeroXAPI:
    __API_URL_BASE = 'https://api.0x.org/'
    __API_VERSION = 'v1'

    def __init__(self, api_base_url=__API_URL_BASE, version=__API_VERSION):
        self.api_base_url = api_base_url
        self.version = version
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

    def get_tokens(self):
        api_url = f'{self.api_base_url}swap/{self.version}/tokens'

        dct_response = self.__request(api_url)
        return pd.DataFrame(dct_response['records'])

    def get_price(self, sell_token, buy_token, sell_amount):
        api_url = f'{self.api_base_url}swap/{self.version}/price'
        api_url = self.__api_url_params(
            api_url, {'sellToken': sell_token, 'buyToken': buy_token, 'sellAmount': sell_amount})

        dct_response = self.__request(api_url)
        return dct_response
