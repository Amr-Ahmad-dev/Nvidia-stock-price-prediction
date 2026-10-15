


import yfinance as yf
import pandas as pd


class DataAcquisition:
    def __init__(self, ticker: str = "NVDA", period: str = "730d", interval: str = "1h"):
      self.ticker = ticker
      self.period = period
      self.interval = interval



    def download_data(self):
        """
        Download historical stock data for the specified ticker using yfinance.

        Returns:
            pd.DataFrame: A DataFrame containing the historical stock data.
        """

        data = yf.download(self.ticker, period=self.period, interval=self.interval,auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        return data