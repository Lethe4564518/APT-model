import os
import sys
import numpy as np
import pandas as pd
import pandas_ta as ta
# import torch
import matplotlib.pyplot as plt
import time

start_time = time.time()

## 嘗試取得當前 HW3.py 檔案的路徑並確保其與 data.csv 在同一資料夾
try:
    current_folder = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_folder, "data.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到文件: {data_path}，請確認 {data_path} 是否與 HW3.py 在同一資料夾")
        
    with open(data_path, 'r', encoding='utf-8') as file:
        data = pd.read_csv(file)
    print(f'\n----- 資料讀取成功 -----\n資料總筆數: {len(data)} 筆\n')

except Exception as e:
    print(f'讀取資料時出錯: {str(e)}')
    sys.exit(1)

## 若CUDA啟用時使用GPU運算，反之則用CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'----- 當前使用設備: {device} -----\n')

## 確保"年月"為datetime格式，先以股票代碼初步排序並用年月再次排序並備份
data['年月'] = pd.to_datetime(data['年月'])
data = data.sort_values(by=['股票代碼', '年月'])
data['股票代碼_備份'] = data['股票代碼']


def calculate_ta(group):
    """ 針對每一檔股票分別計算技術分析相關指標 """
    group["SMA_6"] = group["收盤價"].rolling(window=6).mean()      # 6個月移動平均線
    group["SMA_12"] = group["收盤價"].rolling(window=12).mean()    # 12個月移動平均線

    macd = ta.macd(group["收盤價"], fast=6, slow=12, signal=9)     # MACD
    group["MACD"] = macd["MACD_6_12_9"]
    group["MACD_signal"] = macd["MACDs_6_12_9"]
    group["MACD_hist"] = macd["MACDh_6_12_9"]

    group["RSI_14"] = ta.rsi(group["收盤價"], length=14)           # RSI

    bb = ta.bbands(group["收盤價"], length=12, std=2)              # 布林通道
    group["BB_Upper"] = bb["BBU_12_2.0"]
    group["BB_Lower"] = bb["BBL_12_2.0"]

    group["ATR_14"] = ta.atr(group["最高價"], group["最低價"], group["收盤價"], length=14)    # ATR波動率

    group["52W_High"] = group["收盤價"].rolling(window=12).max()   # 52週新高點
    group["52W_Low"] = group["收盤價"].rolling(window=12).min()    # 52週新低點

    group["OBV"] = ta.obv(group["收盤價"], group["成交量"])        # OBV

    return group

def process_data(data):
    """ 資料處理 """
    try:
        # 確保股票代碼不會同時作為索引和欄位
        data = data.reset_index(drop=True)
        
        # 計算技術指標
        grouped = data.groupby('股票代碼')
        result = pd.concat([calculate_ta(group) for name, group in grouped])
        
        # 向前後填補缺失值
        for col in result.columns:
            if col not in ['股票代碼', '股票名稱', '年月', '股票代碼_備份']:
                result[col] = result.groupby('股票代碼')[col].transform(lambda x: x.ffill().bfill())
        
        result = result.sort_values(by=['股票代碼', '年月'])    # 確保排序
        return result
        
    except Exception as e:
        print(f'資料處理過程中發生錯誤: {str(e)}')
        return None


try:
    # 處理資料
    processed_data = process_data(data)
    if processed_data is None:
        raise ValueError("資料處理失敗")
    
    data = processed_data                   # 更新資料
    
    missing_values = data.isna().sum()      # 檢查缺失值
    if missing_values.any():
        print('----- 資料中仍有缺失值存在 -----')
        print(missing_values[missing_values > 0])
    else:
        print('----- 已確認資料中沒有缺失值 -----\n')
        
except Exception as e:
    print(f'程式執行錯誤: {str(e)}')
    sys.exit(1)




## ============================= 實作開始 ============================= ##
class Splitter:
    def __init__(self, df, time_col: str = '年月'):
        """
        初始化物件
        df: 包含所有資料的DataFrame
        time_col: 日期欄位
        """
        self.df = df.copy()
        self.time_col = time_col

    def split_by_time(self, train_end_time: str):
        """
        根據時間劃分訓練集與測試集
        train_end_time: 訓練集結束的時間(含該期), 測試集則從下一期開始
        return出(train_df, test_df)
        """
        train_df = self.df[self.df[self.time_col] <= pd.to_datetime(train_end_time)]      # 分割出訓練集
        test_df = self.df[self.df[self.time_col] > pd.to_datetime(train_end_time)]        # 分割出測試集
        return train_df, test_df


if __name__ == '__main__':
    panel = Splitter(data)                                              # 分割data
    train_time_df, test_time_df = panel.split_by_time('2019-12-01')     # 按照時間分割
    print('----- 依時間分割訓練集與測試集 -----')
    print(f'訓練集: {len(train_time_df)} 測試集: {len(test_time_df)}\n')




## =============== 第1部份 =============== ##
class APTModel:
    def __init__(self, df: pd.DataFrame, K: int = 8, eigenvectors: np.ndarray = None):
        """
        初始化APT模型
        df: DataFrame
        K: 因子數量, 預設為8
        eigenvectors: 預設為None
        """
        self.df = df
        self.K = K
        self.returns_matrix = None
        self.returns_demeaned = None
        self.eigenvectors = eigenvectors
        self.ft = None
        self.mu_f = None
        self.sigma_f = None
        self.optimal_weights = None
        self.Rp = None
        self.excluded_columns = ['股票代碼', '股票名稱', '年月', '月報酬率', '股票代碼_備份']
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 清理數據，去除無需計算的屬性 """
        return df.drop(columns=self.excluded_columns, errors='ignore')
        
    def prepare_data(self):
        """ 準備並提取數據 """
        self.returns_matrix = self.df.pivot(
            index='年月', 
            columns='股票代碼', 
            values='月報酬率'
        ).values
        self.returns_demeaned = self.returns_matrix - np.mean(self.returns_matrix, axis=0)      # 將月報酬率減去月報酬率的平均值
        self.df = self.clean_data(self.df)
        
    def extract_factors(self, eigenvectors: np.ndarray = None):
        """
        使用PCA提取因子
        預設為會計算eigenvectors, 若輸入已知的eigenvectors則會直接套用
        """
        if eigenvectors is not None:
            self.eigenvectors = eigenvectors
            selected_eigenvectors = self.eigenvectors[:, -self.K:]
        else:
            cov_matrix = np.cov(self.returns_demeaned.T)
            eigenvalues, self.eigenvectors = np.linalg.eigh(cov_matrix)
            selected_eigenvectors = self.eigenvectors[:, -self.K:]
        
        # 確保不管 K 為何, selected_eigenvectors 都是二維的
        if selected_eigenvectors.ndim == 1:
            selected_eigenvectors = selected_eigenvectors.reshape(-1, 1)
        
        self.ft = np.dot(self.returns_matrix, selected_eigenvectors)
        return selected_eigenvectors
        
    def calculate_factor_moments(self):
        """ 計算因子的平均及covariance矩陣 """
        self.mu_f = np.mean(self.ft, axis=0)
        # 確保 ft 是二維的
        if self.ft.ndim == 1:
            self.ft = self.ft.reshape(-1, 1)
        self.sigma_f = np.cov(self.ft.T)
        # 確保 sigma_f 是二維的
        if self.sigma_f.ndim == 0:
            self.sigma_f = np.array([[self.sigma_f]])
        
    def calculate_optimal_portfolio(self):
        """ 按照公式計算最適投資組合 """
        sigma_f_inv = np.linalg.inv(self.sigma_f)
        ones_vector = np.ones(self.K).reshape(-1, 1) if self.K == 1 else np.ones(self.K)    # 確保 ones_vector 的維度正確
        numerator = np.dot(sigma_f_inv, self.mu_f)
        denominator = np.dot(ones_vector.T, numerator)
        self.optimal_weights = numerator / denominator
        self.Rp = np.dot(self.ft, self.optimal_weights)
        
    def get_performance(self):
        """ 計算 Mean-to-volatility ratio """
        portfolio_mean = np.mean(self.Rp)               # 計算平均報酬率
        portfolio_std = np.std(self.Rp)                 # 計算波動率   
        mtv_ratio = portfolio_mean / portfolio_std      # 計算 Mean-to-volatility ratio
        return mtv_ratio
    
    def fit(self):
        """ 執行 APT model """
        self.prepare_data()
        self.extract_factors()
        self.calculate_factor_moments()
        self.calculate_optimal_portfolio()
        return self


if __name__ == '__main__':
    # 使用訓練集輸出 APT model 結果
    apt_model_train = APTModel(train_time_df)
    apt_model_train.fit()
    selected_eigenvectors = apt_model_train.eigenvectors[:, -apt_model_train.K:]        # 儲存要用於第2部份的 eigenvectors
    

    print('========== 第 1 部份 ==========')
    print('(a) 因子平均報酬率 μf:')
    print(apt_model_train.mu_f)
    
    print('\n(b) 因子cov矩陣 Σf:')
    print(apt_model_train.sigma_f)
    
    print('\n(c) 最適投資組合權重:')
    print(apt_model_train.optimal_weights)
    
    mtv_ratio = apt_model_train.get_performance()
    print('\n(d) Mean-to-volatility ratio:')
    print(f'Mean-to-volatility ratio: {mtv_ratio:.4f}\n')




## =============== 第2部份 =============== ##
if __name__ == '__main__':
    # 使用訓練集的 eigenvectors 在測試集進行計算
    apt_model_test = APTModel(test_time_df, eigenvectors=selected_eigenvectors)
    apt_model_test.fit()

    print('========== 第 2 部份 ==========')
    print('(a) 因子平均報酬率 μf:')
    print(apt_model_test.mu_f)
    
    print('\n(b) 因子cov矩陣 Σf:')
    print(apt_model_test.sigma_f)
    
    print('\n(c) 最適投資組合權重:')
    print(apt_model_test.optimal_weights)
    
    mtv_ratio = apt_model_test.get_performance()
    print('\n(d) Mean-to-volatility ratio:')
    print(f'Mean-to-volatility ratio: {mtv_ratio:.4f}\n')

    


## =============== 第3部份 =============== ##
if __name__ == '__main__':
    print('========== 第 3 部份 ==========')

    max_K = 7                   # 因子數量最大值(hyperparameter)
    train_mtv_ratios = []       # 儲存訓練集的 MTV ratio
    test_mtv_ratios = []        # 儲存測試集的 MTV ratio
    eigenvectors_set = []       # 儲存訓練集的 eigenvectors
    for K in range(1, max_K+1):
        apt_model_i = APTModel(train_time_df, K=K)
        apt_model_i.fit()
        eigenvectors_set.append(apt_model_i.eigenvectors[:, -apt_model_i.K:])
        mtv_ratio = apt_model_i.get_performance()
        train_mtv_ratios.append(mtv_ratio)
        print(f'K={K} 時 Mean-to-volatility ratio: {mtv_ratio:.4f}')
    print(f'訓練集最適因子數量 K = {int(train_mtv_ratios.index(max(train_mtv_ratios))) + 1}\n')    # 選出訓練集中最大的 MTV ratio 所對應的因子數量
    
    for k in range(1, max_K+1):
        apt_model_j = APTModel(test_time_df, K=k, eigenvectors = eigenvectors_set[k-1])
        apt_model_j.fit()
        mtv_ratio = apt_model_j.get_performance()
        test_mtv_ratios.append(mtv_ratio)
        print(f'K={k} 時 Mean-to-volatility ratio: {mtv_ratio:.4f}')
    print(f'測試集最適因子數量 K = {int(test_mtv_ratios.index(max(test_mtv_ratios))) + 1}\n')     # 選出測試集中最大的 MTV ratio 所對應的因子數量

    end_time = time.time()
    print(f'運行總耗時: {end_time - start_time:.4f} 秒\n')

    # 繪製訓練集與測試集的 MTV ratio 和不同因子數量 K 的關係圖
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_K+1), train_mtv_ratios, 'b-o', label='Training Set')
    plt.plot(range(1, max_K+1), test_mtv_ratios, 'r-o', label='Testing Set')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Mean-to-volatility Ratio')
    plt.title('Mean-to-volatility Ratio vs Number of Factors')
    plt.legend()
    plt.grid(True)
    plt.show()

    







