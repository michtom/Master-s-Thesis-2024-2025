import asyncio
import datetime

import aiohttp
import logging
from typing import List
import pandas as pd

BASE_ENDPOINT = 'https://api.coingecko.com/api/v3/coins'
COINGECKO_SYMBOL_MAPPER = {'bitcoin': 'BTC', 'ethereum': 'ETH', 'solana': 'SOL'}


async def fetch_coingecko_current_data(symbols: List[str], path: str) -> dict:
    logging.info('fetching market data from coingecko')
    tasks_symbol_data = [fetch_data_for_one_symbol(symbol) for symbol in symbols]
    symbols_data = await asyncio.gather(*tasks_symbol_data)
    result_dict = {symbol: symbol_data for symbol, symbol_data in zip(symbols, symbols_data)}
    dfs_general = []
    timestamp = datetime.datetime.now()
    for symbol, data in result_dict.items():
        df = parse_coingecko_dict(data)
        df['symbol'] = COINGECKO_SYMBOL_MAPPER.get(symbol, symbol)
        df['timestamp'] = timestamp
        dfs_general.append(df)
    df_general = pd.concat(dfs_general)
    df_general.to_csv(f"{path}/general_market_data_{timestamp.strftime('%Y-%m-%d-%H')}.csv", index=False)
    logging.info('successfully fetched general market data')
    tasks_market_chart = [fetch_market_chart_for_one_symbol(symbol) for symbol in symbols]
    market_chart_data = await asyncio.gather(*tasks_market_chart)
    results_market_chart = {symbol: symbol_market for symbol, symbol_market in zip(symbols, market_chart_data)}
    dfs_market_chart = []
    for symbol, data in results_market_chart.items():
        df = parse_market_chart(data)
        df['symbol'] = COINGECKO_SYMBOL_MAPPER.get(symbol, symbol)
        dfs_market_chart.append(df)
    market_chart_df = pd.concat(dfs_market_chart)
    market_chart_df.to_csv(f"{path}/market_chart_data_{timestamp.strftime('%Y-%m-%d-%H')}.csv", index=False)
    logging.info('successfully fetched market chart data')
    return result_dict


def parse_coingecko_dict(coingecko_dict: dict) -> pd.DataFrame:
    market_data = coingecko_dict.get('market_data', {})
    current_price = market_data.get('current_price', {}).get('usd')
    total_volume = market_data.get('total_volume', {}).get('usd')
    fully_diluted_valuation = market_data.get('fully_diluted_valuation', {}).get('usd')
    high_24h = market_data.get('high_24h', {}).get('usd')
    low_24h = market_data.get('low_24h', {}).get('usd')
    price_change_24h = market_data.get('price_change_24h')
    price_change_percentage_24h = market_data.get('price_change_percentage_24h')
    price_change_percentage_7d = market_data.get('price_change_percentage_7d')
    circulating_supply = market_data.get('circulating_supply')
    df = pd.DataFrame({'current_price': [current_price], 'total_volume': [total_volume],
                       'fully_diluted_valuation': [fully_diluted_valuation], 'high_24h': [high_24h], 'low_24h': [low_24h],
                       'price_change_24h': [price_change_24h],
                       'price_change_percentage_24h': [price_change_percentage_24h],
                       'price_change_percentage_7d': [price_change_percentage_7d],
                       'circulating_supply': [circulating_supply]})
    return df


async def fetch_market_chart_for_one_symbol(symbol: str) -> dict:
    params = {
        "vs_currency": 'usd',
        "days": "1",
    }
    for attempt in range(2):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f'{BASE_ENDPOINT}/{symbol}/market_chart', params=params) as resp:
                    if resp.status == 429:
                        logging.info('rate limits reached, sleeping for 1 minute')
                        await asyncio.sleep(60)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                logging.error(f'error while fetching data for {symbol}: {e}')
        break
    return {}


async def fetch_data_for_one_symbol(symbol: str) -> dict:
    for attempt in range(2):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f'{BASE_ENDPOINT}/{symbol}') as response:
                    if response.status == 429:
                        logging.info('rate limits reached, sleeping for 1 minute')
                        await asyncio.sleep(60)
                        continue
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                logging.error(f'error while fetching data for {symbol}: {e}')
        break
    return {}


def parse_market_chart(data: dict) -> pd.DataFrame:
    prices_dict = data.get('prices', [])
    df_prices = pd.DataFrame(prices_dict, columns=['timestamp', 'price'])
    total_volume_dict = data.get('total_volumes', {})
    df_total_volume = pd.DataFrame(total_volume_dict, columns=['timestamp', 'volume'])
    market_cap_dict = data.get('market_caps', {})
    df_market_cap = pd.DataFrame(market_cap_dict, columns=['timestamp', 'market_cap'])
    df = df_prices.merge(df_total_volume, on='timestamp').merge(df_market_cap, on='timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    folder_path = '../../data/market_data'
    symbols = ['bitcoin', 'ethereum', 'solana']
    asyncio.run(fetch_coingecko_current_data(symbols, folder_path))
