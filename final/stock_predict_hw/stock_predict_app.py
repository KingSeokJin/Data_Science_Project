import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import math
import random
import ssl
import requests
from io import StringIO

# ---------------------------------------------------------
# 설정 (Configuration) 
# ---------------------------------------------------------
CONFIG = {
    # tickers는 main 함수에서 동적으로 할당됨
    'start_date': '2021-01-01', # 학습 시작 기간
    'end_date': '2025-11-30',  # 학습 끝 기간
    'seq_length': 20, # 20일치 데이터를 한 묶음으로 봄      
    'input_dim_seq': 5,       # Open, High, Low, Close, Volume
    'input_dim_static': 3,    # PER, PBR, ROE
    'hidden_dim': 64,         
    'output_dim': 1,          
    'num_layers': 2,
    'nhead': 4,               
    'epochs': 50,             
    'batch_size': 32,         # batch size
    'learning_rate': 0.001,
    'train_ratio': 0.8,     # train 0.8 test 0.2  
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Using device: {CONFIG['device']}")

# ---------------------------------------------------------
# 0. 유틸리티: S&P 500 IT 섹터 종목 가져오기
# ---------------------------------------------------------
def get_sp500_it_tickers():
    """위키피디아에서 S&P 500 종목 리스트를 크롤링하여 IT 섹터만 필터링"""
    print("\n[System] S&P 500 IT 섹터 종목 리스트를 조회 중입니다...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        # 봇 차단을 우회하기 위한 헤더 설정 (브라우저인로 위장)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # requests로 HTML 가져오기
        response = requests.get(url, headers=headers)
        response.raise_for_status() # 에러 발생 시 예외 처리
        
        # HTML 텍스트를 파싱
        tables = pd.read_html(StringIO(response.text))
        df = tables[0] # 첫 번째 테이블이 종목 리스트
        
        # 'GICS Sector'가 'Information Technology'인 종목 필터링
        it_sector_df = df[df['GICS Sector'] == 'Information Technology']
        tickers = it_sector_df['Symbol'].tolist()
        
        # yfinance용 티커 포맷 변환 (예: BRK.B -> BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f"[System] 총 {len(tickers)}개의 IT 섹터 종목을 식별했습니다.")
        return tickers
        
    except Exception as e:
        print(f"[Error] 종목 리스트를 가져오는 데 실패했습니다: {e}")
        print("[System] 기본 대표 종목 10개로 대체합니다.")
        return [
            'AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 
            'CRM', 'ADBE', 'ORCL', 'CSCO', 'ACN'
        ]

# ---------------------------------------------------------
# 1. 데이터셋 클래스 (Hybrid Dataset)
# ---------------------------------------------------------
class HybridStockDataset(Dataset):
    def __init__(self, seq_data, static_data, targets):
        self.seq_data = torch.tensor(seq_data, dtype=torch.float32)
        self.static_data = torch.tensor(static_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.seq_data[idx], self.static_data[idx], self.targets[idx]

# ---------------------------------------------------------
# 2. 데이터 처리 및 유틸리티
# ---------------------------------------------------------
class DataManager:
    def __init__(self, tickers, start, end, seq_length):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.seq_length = seq_length
        self.scaler_seq = MinMaxScaler()
        self.scaler_static = MinMaxScaler()
        self.scaler_target = MinMaxScaler() 

    def get_fundamentals(self, ticker):
        """yfinance에서 PER, PBR, ROE 가져오기"""
        try:
            info = yf.Ticker(ticker).info
            per = info.get('trailingPE', 0)
            pbr = info.get('priceToBook', 0)
            roe = info.get('returnOnEquity', 0)
            
            # None 타입 체크 및 0 대체
            per = per if per is not None else 0
            pbr = pbr if pbr is not None else 0
            roe = roe if roe is not None else 0
            
            return [per, pbr, roe]
        except:
            return [0, 0, 0]

    def prepare_data(self):
        print("\n[System] 데이터 다운로드 및 전처리 시작 (시간이 소요될 수 있습니다)...")
        
        # 1. 종목 분할 (8:2)
        random.seed(42)
        shuffled_tickers = self.tickers.copy()
        random.shuffle(shuffled_tickers)
        split_idx = int(len(shuffled_tickers) * CONFIG['train_ratio']) # train_ratio = 0.8
        train_tickers = shuffled_tickers[:split_idx]
        test_tickers = shuffled_tickers[split_idx:]
        
        print(f"학습용 종목 수: {len(train_tickers)}개")
        print(f"테스트용 종목 수: {len(test_tickers)}개") 

        # 데이터 저장소
        train_raw_seq, train_raw_static, train_raw_y = [], [], []
        self.stock_data_map = {} 

        # 2. 데이터 다운로드 및 수집
        total_tickers = len(self.tickers)
        for idx, ticker in enumerate(self.tickers):
            if (idx + 1) % 5 == 0:
                print(f"Processing... ({idx + 1}/{total_tickers})")

            # 2-1. 시계열 데이터
            df = yf.download(ticker, start=self.start, end=self.end, progress=False)
            if len(df) < self.seq_length + 1:
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker, axis=1, level=1)

            seq_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            if seq_df.empty: continue
            
            target_series = seq_df['Close'].values
            seq_values = seq_df.values

            # 2-2. 고정 데이터
            fundamentals = self.get_fundamentals(ticker)
            
            # 해당 ticker가 학습용 리스트에 있는지 확인
            is_train = ticker in train_tickers
            
            sequences = []
            targets = []
            statics = [] 
            
            for i in range(len(seq_values) - self.seq_length):
                sequences.append(seq_values[i : i+self.seq_length])
                targets.append(target_series[i + self.seq_length])
                statics.append(fundamentals)
            
            if is_train: # 학습용 종목일 때만 학습 데이터 리스트에 추가합니다.
                train_raw_seq.extend(sequences)
                train_raw_y.extend(targets)
                train_raw_static.extend(statics)
            else:
                # 테스트용 종목은 별도의 딕셔너리(stock_data_map)에 따로 저장해둡니다.
                self.stock_data_map[ticker] = {
                    'seq': np.array(sequences),
                    'static': np.array(statics),
                    'target': np.array(targets)
                }

        print("[System] 데이터 다운로드 완료. 정규화 수행 중...")

        # 3. 정규화 (MinMaxScaler)
        train_raw_seq = np.array(train_raw_seq)
        train_raw_static = np.array(train_raw_static)
        train_raw_y = np.array(train_raw_y).reshape(-1, 1)

        N, L, D = train_raw_seq.shape
        self.scaler_seq.fit(train_raw_seq.reshape(-1, D))
        self.scaler_static.fit(train_raw_static)
        self.scaler_target.fit(train_raw_y)

        train_seq_scaled = self.scaler_seq.transform(train_raw_seq.reshape(-1, D)).reshape(N, L, D)
        train_static_scaled = self.scaler_static.transform(train_raw_static)
        train_y_scaled = self.scaler_target.transform(train_raw_y)

        test_datasets = {} 
        for ticker, data in self.stock_data_map.items():
            if len(data['seq']) == 0: continue
            
            t_N, t_L, t_D = data['seq'].shape
            s_seq = self.scaler_seq.transform(data['seq'].reshape(-1, t_D)).reshape(t_N, t_L, t_D)
            s_static = self.scaler_static.transform(data['static'])
            s_y = self.scaler_target.transform(data['target'].reshape(-1, 1))
            
            test_datasets[ticker] = HybridStockDataset(s_seq, s_static, s_y)

        train_dataset = HybridStockDataset(train_seq_scaled, train_static_scaled, train_y_scaled)
        
        print(f"[System] 준비 완료. 총 학습 샘플 수: {len(train_dataset)}")
        return train_dataset, test_datasets, test_tickers

# ---------------------------------------------------------
# 3. 모델 정의 (Hybrid Architecture)
# ---------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, model_type, config):
        super(HybridModel, self).__init__()
        self.model_type = model_type
        
        # Sequence Branch
        if model_type == 'RNN':
            self.seq_layer = nn.RNN(
                input_size=config['input_dim_seq'],
                hidden_size=config['hidden_dim'],
                num_layers=config['num_layers'],
                batch_first=True, dropout=0.2
            )
        elif model_type == 'LSTM':
            self.seq_layer = nn.LSTM(
                input_size=config['input_dim_seq'],
                hidden_size=config['hidden_dim'],
                num_layers=config['num_layers'],
                batch_first=True, dropout=0.2
            )
        elif model_type == 'Transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config['hidden_dim'],
                nhead=config['nhead'],
                dim_feedforward=config['hidden_dim']*4,
                dropout=0.1, batch_first=True
            )
            self.seq_layer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
            self.seq_embedding = nn.Linear(config['input_dim_seq'], config['hidden_dim']) 

        # Static Branch
        self.static_layer = nn.Sequential(
            nn.Linear(config['input_dim_static'], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # Concatenation & Output
        self.fc_final = nn.Linear(config['hidden_dim'] + 8, config['output_dim'])

    def forward(self, x_seq, x_static):
        if self.model_type == 'Transformer':
            x_seq = self.seq_embedding(x_seq) 
            out = self.seq_layer(x_seq)
            seq_feat = out[:, -1, :] 
        else:
            out, _ = self.seq_layer(x_seq)
            seq_feat = out[:, -1, :] 

        static_feat = self.static_layer(x_static)
        combined = torch.cat((seq_feat, static_feat), dim=1)
        output = self.fc_final(combined)
        return output

# ---------------------------------------------------------
# 4. 학습 및 평가 루프
# ---------------------------------------------------------
def train_and_evaluate(model_type, train_loader, test_datasets, test_tickers, config, scaler_target):
    model = HybridModel(model_type, config).to(config['device'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"\n>>> [{model_type}] 모델 학습 시작...")
    start_time = time.time()
    loss_history = []

    model.train()
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for seq_batch, static_batch, target_batch in train_loader:
            seq_batch = seq_batch.to(config['device'])
            static_batch = static_batch.to(config['device'])
            target_batch = target_batch.to(config['device'])

            optimizer.zero_grad()
            outputs = model(seq_batch, static_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.6f}")

    train_time = time.time() - start_time
    print(f"[{model_type}] 학습 완료. 소요 시간: {train_time:.2f}초")

    results = {}
    model.eval()
    
    # 실제 데이터가 존재하는 테스트 종목만 필터링
    valid_test_tickers = [t for t in test_tickers if t in test_datasets] # 테스트 종목에서 골라서 담아두기
    # 아래서 테스트용 종목 중 2개를 선정. 이 selected_test_tickers는 위에서 확인했듯 학습 데이터에 포함되지 않았던 종목들
    if len(valid_test_tickers) < 2: 
        selected_test_tickers = valid_test_tickers
    else:
        selected_test_tickers = valid_test_tickers[:2]
    
    with torch.no_grad():
        for ticker in selected_test_tickers:
            # 아까 따로 빼둔 test_datasets에서 해당 종목의 데이터를 가져오기.
            ds = test_datasets[ticker]
            seq_data = ds.seq_data.to(config['device'])
            static_data = ds.static_data.to(config['device'])
            real_y = ds.targets.numpy()
            
            pred_y = model(seq_data, static_data).cpu().numpy()
            
            pred_price = scaler_target.inverse_transform(pred_y)
            real_price = scaler_target.inverse_transform(real_y)
            
            results[ticker] = (real_price, pred_price)

    return {
        'model_type': model_type,
        'train_time': train_time,
        'final_loss': loss_history[-1],
        'loss_history': loss_history,
        'predictions': results
    }

# ---------------------------------------------------------
# 5. 메인 실행
# ---------------------------------------------------------
def main():
    # S&P 500 IT 섹터 종목 전체 가져오기
    it_tickers = get_sp500_it_tickers()
    
    # 데이터 준비
    manager = DataManager(it_tickers, CONFIG['start_date'], CONFIG['end_date'], CONFIG['seq_length'])
    train_dataset, test_datasets, test_tickers = manager.prepare_data()
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    models = ['RNN', 'LSTM', 'Transformer']
    summary = []

    plt.figure(figsize=(10, 5))

    for m_type in models:
        res = train_and_evaluate(m_type, train_loader, test_datasets, test_tickers, CONFIG, manager.scaler_target)
        summary.append(res)
        plt.plot(res['loss_history'], label=f"{m_type} Loss")

    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    print("\n" + "="*60)
    print("Model Performance Comparison (All S&P 500 IT Stocks)")
    print("="*60)
    print(f"{'Model':<12} | {'Time (sec)':<10} | {'Final Loss':<10}")
    print("-" * 60)
    for res in summary:
        print(f"{res['model_type']:<12} | {res['train_time']:<10.2f} | {res['final_loss']:.6f}")
    print("="*60)

    # 테스트 결과 시각화
    if summary and summary[0]['predictions']:
        test_tickers_to_plot = list(summary[0]['predictions'].keys())
        
        for ticker in test_tickers_to_plot:
            plt.figure(figsize=(12, 6))
            
            real_price = summary[0]['predictions'][ticker][0]
            plt.plot(real_price, label='Real Price', color='black', linewidth=2)
            
            colors = ['red', 'green', 'blue']
            for i, res in enumerate(summary):
                if ticker in res['predictions']:
                    pred_price = res['predictions'][ticker][1]
                    plt.plot(pred_price, label=f'{res["model_type"]} Prediction', color=colors[i], linestyle='--')
                
            plt.title(f"Test Result: {ticker}")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.show()
    else:
        print("테스트할 데이터가 부족하여 그래프를 그릴 수 없습니다.")

if __name__ == "__main__":
    main()