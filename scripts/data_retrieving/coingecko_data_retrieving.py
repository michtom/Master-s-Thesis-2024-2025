import asyncio
import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp
import logging
from typing import List, Any
import pandas as pd

BASE_ENDPOINT = 'https://api.coingecko.com/api/v3/coins'
COINGECKO_SYMBOL_MAPPER = {'bitcoin': 'BTC', 'ethereum': 'ETH', 'solana': 'SOL'}


async def fetch_coingecko_current_data(symbols: List[str], path: str) -> dict:
    logging.info('fetching market data from coingecko')
    tasks_symbol_data = [fetch_data_for_one_symbol(symbol) for symbol in symbols]
    symbols_data = await asyncio.gather(*tasks_symbol_data)
    result_dict = {symbol: symbol_data for symbol, symbol_data in zip(symbols, symbols_data)}
    parsed_results = []
    timestamp = datetime.datetime.now()
    for symbol, data in result_dict.items():
        symbol = COINGECKO_SYMBOL_MAPPER.get(symbol, symbol)
        parsed_result = parse_coingecko_dict(data, symbol)
        parsed_result['symbol'] = symbol
        parsed_result['timestamp'] = timestamp
        parsed_results.append(parsed_result)
    df = pd.DataFrame(parsed_results)
    df.to_csv(f"{path}/market_chart_data_{timestamp.strftime('%Y-%m-%d-%H')}.csv", index=False)
    logging.info('successfully fetched market data')
    return result_dict


def parse_coingecko_dict(coingecko_dict: dict, symbol: str) -> dict[str, Any]:
    tickers = coingecko_dict.get('tickers', [])
    tickers = [ticker for ticker in tickers if ticker.get('base', '') == symbol and ticker.get('target', '') == 'USDT']
    if len(tickers) == 0:
        logging.error(f"no data for {symbol}-USDT on binance in coingecko")
        return {}
    if len(tickers) >= 1:
        logging.warning(f"found more than one match for {symbol}-USDT on binance in coingecko, selecting the first one")
    ticker = tickers[0]
    price = ticker.get('last')
    volume = ticker.get('volume')
    return {'volume': volume, 'price': price}

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=20))
async def fetch_data_for_one_symbol(symbol: str) -> dict:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f'{BASE_ENDPOINT}/{symbol}/tickers?exchange_ids=binance') as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logging.error(f'error while fetching data for {symbol}: {e}')
            raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    folder_path = '../data/market_data'
    symbols = ['bitcoin', 'ethereum', 'solana']
    asyncio.run(fetch_coingecko_current_data(symbols, folder_path))
