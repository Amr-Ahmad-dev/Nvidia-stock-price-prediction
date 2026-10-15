import Data_acquisition as da
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data=da.DataAcquisition( num_of_inputs_rows=5).download_data()
#data.to_csv("data.csv", index=False)

class data_visualization:
    def __init__(self, data: pd.DataFrame=None):
        self.data = data or None
    def Visualize (df):
        """
        Visualizes the stock data by plotting the closing price and z-score.

        Args: df (pd.DataFrame): A DataFrame containing the stock data with 'Close' and 'z_score' columns.
        """
        
        plt.subplot(2, 1, 1)
        plt.plot(df['Datetime'].to_numpy(), df[['Close','Open',"High","Low"]].to_numpy(), label='Close Price', color='blue')
        plt.title('Stock Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)

        subplot = plt.subplot(2, 1, 2)

        plt.plot(df['Datetime'].to_numpy(), df['z_score'].to_numpy(), label='Z-Score', color='orange')
        plt.title('Z-Score of Price Range')
        plt.tight_layout()
    
        plt.show()

        corr_2 = df[['Close','Open','z_score','range','Volume',"High","Low" ]].corr()

        sns.heatmap(
        corr_2,
        #mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()



        c_n=df.columns
        for i in c_n:
            if "lag" in i:
                continue
            if i in ['Datetime',]:
                continue
            plt.figure(figsize=(10, 4))
            sns.lineplot(x='Datetime', y=i, data=df)
            plt.title(f'{i} over Time')
            plt.xlabel('Date')
            plt.ylabel(i)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            sns.histplot(x=i, data=df, kde=True)
            plt.title(f'Distribution of {i}')
            plt.xlabel(i)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()
        
        return
    def conf(cm):

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"]
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()




class DataPreprocessing:

    def __init__(
        self,
        data: pd.DataFrame,
        window=24,
        lage: int = 6
    ):

        self.data = data.copy()
        self.window_size = window
        self.lage = lage

    def preprocess_data(self):

        # =====================================================
        # BASIC DATA PREPARATION
        # =====================================================

        self.data['Datetime'] = pd.to_datetime(
            self.data['Datetime'],
            utc=True
        )

        self.data.sort_values(
            by='Datetime',
            inplace=True
        )

        self.data.reset_index(
            drop=True,
            inplace=True
        )

        # Remove duplicate timestamps
        self.data.drop_duplicates(
            subset=['Datetime'],
            inplace=True
        )

        # =====================================================
        # BASIC RETURNS
        # =====================================================

        self.data["return"] = (
            self.data["Close"].pct_change()
        )

        self.data["return_3"] = (
            self.data["Close"].pct_change(3)
        )

        self.data["return_6"] = (
            self.data["Close"].pct_change(6)
        )

        self.data["return_12"] = (
            self.data["Close"].pct_change(12)
        )

        self.data["return_24"] = (
            self.data["Close"].pct_change(24)
        )

        # =====================================================
        # CANDLE / PRICE STRUCTURE
        # =====================================================

        self.data["range"] = (
            (self.data["High"] - self.data["Low"])
            / self.data["Open"]
        )

        self.data["body"] = (
            (self.data["Close"] - self.data["Open"])
            / self.data["Open"]
        )

        self.data["upper_wick"] = (
            (
                self.data["High"]
                - self.data[["Open", "Close"]].max(axis=1)
            )
            / self.data["Open"]
        )

        self.data["lower_wick"] = (
            (
                self.data[["Open", "Close"]].min(axis=1)
                - self.data["Low"]
            )
            / self.data["Open"]
        )

        # =====================================================
        # CURRENT 24-PERIOD Z-SCORE
        # =====================================================

        self.data["z_score"] = (
            self.data["range"]
            .rolling(window=self.window_size)
            .apply(
                lambda x:
                    (
                        (x.iloc[-1] - x.mean()) / x.std()
                        if x.std() != 0
                        else 0
                    )
            )
        )

        # =====================================================
        # TARGET
        #
        # IMPORTANT:
        # This uses the CURRENT z_score.
        # Predictors below will use previous observations.
        # =====================================================

        self.data["volatile"] = (
            self.data["z_score"]
            .abs()
            .apply(
                lambda x: 1 if x > 0.5 else 0
            )
        )

        # =====================================================
        # ROLLING VOLATILITY
        # =====================================================

        self.data["volatility_5"] = (
            self.data["return"]
            .rolling(5)
            .std()
        )

        self.data["volatility_12"] = (
            self.data["return"]
            .rolling(12)
            .std()
        )

        self.data["volatility_24"] = (
            self.data["return"]
            .rolling(24)
            .std()
        )

        self.data["volatility_48"] = (
            self.data["return"]
            .rolling(48)
            .std()
        )

        # =====================================================
        # RANGE Z-SCORES AT DIFFERENT HORIZONS
        # =====================================================

        for window in [6, 12, 24, 48]:

            mean = (
                self.data["range"]
                .rolling(window)
                .mean()
            )

            std = (
                self.data["range"]
                .rolling(window)
                .std()
            )

            self.data[f"range_z_{window}"] = (
                (self.data["range"] - mean)
                / std.replace(0, np.nan)
            )

        # =====================================================
        # RETURN Z-SCORES
        # =====================================================

        for window in [12, 24, 48]:

            mean = (
                self.data["return"]
                .rolling(window)
                .mean()
            )

            std = (
                self.data["return"]
                .rolling(window)
                .std()
            )

            self.data[f"return_z_{window}"] = (
                (self.data["return"] - mean)
                / std.replace(0, np.nan)
            )

        # =====================================================
        # VOLUME FEATURES
        # =====================================================

        self.data["volume_change"] = (
            self.data["Volume"].pct_change()
        )

        volume_mean = (
            self.data["Volume"]
            .rolling(24)
            .mean()
        )

        volume_std = (
            self.data["Volume"]
            .rolling(24)
            .std()
        )

        self.data["volume_z"] = (
            (
                self.data["Volume"]
                - volume_mean
            )
            / volume_std.replace(0, np.nan)
        )

        # =====================================================
        # MOVING AVERAGE POSITION
        # =====================================================

        self.data["close_ma_6"] = (
            self.data["Close"]
            / self.data["Close"].rolling(6).mean()
            - 1
        )

        self.data["close_ma_12"] = (
            self.data["Close"]
            / self.data["Close"].rolling(12).mean()
            - 1
        )

        self.data["close_ma_24"] = (
            self.data["Close"]
            / self.data["Close"].rolling(24).mean()
            - 1
        )

        # =====================================================
        # WIN / DIRECTION
        # =====================================================

        self.data["win"] = (
            self.data["return"]
            .rolling(window=self.window_size)
            .mean()
            .apply(
                lambda x: 1 if x > 0 else 0
            )
        )

        # =====================================================
        # LAG FEATURES
        # =====================================================

        for i in range(1, self.lage + 1):

            # -------------------------------------------------
            # Original features
            # -------------------------------------------------

            self.data[f"Close_lag_{i}"] = (
                self.data["Close"]
                .shift(i)
            )

            self.data[f"Open_lag_{i}"] = (
                self.data["Open"]
                .shift(i)
            )

            self.data[f"High_lag_{i}"] = (
                self.data["High"]
                .shift(i)
            )

            self.data[f"Low_lag_{i}"] = (
                self.data["Low"]
                .shift(i)
            )

            self.data[f"Volume_lag_{i}"] = (
                self.data["Volume"]
                .shift(i)
            )

            self.data[f"range_lag_{i}"] = (
                self.data["range"]
                .shift(i)
            )

            self.data[f"z_score_lag_{i}"] = (
                self.data["z_score"]
                .shift(i)
            )

            self.data[f"win_lag_{i}"] = (
                self.data["win"]
                .shift(i)
            )

            self.data[f"volatile_lag_{i}"] = (
                self.data["volatile"]
                .shift(i)
            )

            # -------------------------------------------------
            # New relative features
            # -------------------------------------------------

            self.data[f"return_lag_{i}"] = (
                self.data["return"]
                .shift(i)
            )

            self.data[f"body_lag_{i}"] = (
                self.data["body"]
                .shift(i)
            )

            self.data[f"upper_wick_lag_{i}"] = (
                self.data["upper_wick"]
                .shift(i)
            )

            self.data[f"lower_wick_lag_{i}"] = (
                self.data["lower_wick"]
                .shift(i)
            )

            self.data[f"volatility_5_lag_{i}"] = (
                self.data["volatility_5"]
                .shift(i)
            )

            self.data[f"volatility_12_lag_{i}"] = (
                self.data["volatility_12"]
                .shift(i)
            )

            self.data[f"volatility_24_lag_{i}"] = (
                self.data["volatility_24"]
                .shift(i)
            )

            self.data[f"range_z_6_lag_{i}"] = (
                self.data["range_z_6"]
                .shift(i)
            )

            self.data[f"range_z_12_lag_{i}"] = (
                self.data["range_z_12"]
                .shift(i)
            )

            self.data[f"range_z_24_lag_{i}"] = (
                self.data["range_z_24"]
                .shift(i)
            )

            self.data[f"range_z_48_lag_{i}"] = (
                self.data["range_z_48"]
                .shift(i)
            )

            self.data[f"return_z_24_lag_{i}"] = (
                self.data["return_z_24"]
                .shift(i)
            )

            self.data[f"volume_z_lag_{i}"] = (
                self.data["volume_z"]
                .shift(i)
            )

            self.data[f"volume_change_lag_{i}"] = (
                self.data["volume_change"]
                .shift(i)
            )

            self.data[f"close_ma_24_lag_{i}"] = (
                self.data["close_ma_24"]
                .shift(i)
            )

        # =====================================================
        # REMOVE INVALID VALUES
        # =====================================================

        self.data.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        self.data.dropna(
            inplace=True
        )

        self.data.reset_index(
            drop=True,
            inplace=True
        )

        # =====================================================
        # INFORMATION
        # =====================================================

        print(
            "Data preprocessing completed."
        )

        print(
            "Data shape after preprocessing:",
            self.data.shape
        )

        print(
            "Missing values:",
            self.data.isnull().sum().sum()
        )

        print(
            "Class distribution:"
        )

        print(
            self.data["volatile"]
            .value_counts()
        )

        print(
            "Class proportions:"
        )

        print(
            self.data["volatile"]
            .value_counts(normalize=True)
        )

        print(
            "Data types:"
        )

        print(
            self.data.dtypes
        )

        print(
            "First rows:"
        )

        print(
            self.data.head()
        )

        print(
            "Last rows:"
        )

        print(
            self.data.tail()
        )

        print(
            "Summary statistics:"
        )

        print(
            self.data.describe()
        )

        return self.data


if __name__ == '__main__':

    df = pd.read_csv(
        "data_2.csv"
    )

    p = DataPreprocessing(
        data=df,
        lage=6,
        window=24
    )

    p = p.preprocess_data()

    p.to_csv(
        "data_preprocessed.csv",
        index=False
    )
