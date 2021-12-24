import os
import pandas as pd

from etherscan import Etherscan


def get_tx(wallet_id, api_key):
    eth = Etherscan(api_key=api_key)

    if not isinstance(wallet_id, list):
        wallet_id = [wallet_id]

    lst_tx = []
    for w in wallet_id:
        lst_tx += eth.get_normal_txs_by_address(address=w, startblock=12760000, endblock=12900000, sort='true')
    df_tx = pd.DataFrame(lst_tx)
    eth.get_tx_receipt_status()
    return df_tx


if __name__ == '__main__':
    folder, _ = os.path.split(__file__)
    folder = folder[:-4]
    data_folder = os.path.join(folder, 'data')
    with open(os.path.join(data_folder, 'etherscan.txt')) as f:
        my_api_key = f.readlines()[0]
    my_wallet = pd.read_csv(os.path.join(data_folder, 'my_wallet.txt'), header=None)[0].to_list()

    get_tx(my_wallet, my_api_key)
