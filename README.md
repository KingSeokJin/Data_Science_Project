# Data Science : Clustering & Stock Prediction

이 레포지토리는 데이터 사이언스 과목의 주요 과업인 **비지도 학습(Clustering) 알고리즘 분석** 및 **딥러닝 기반 시계열 주가 예측** 실습 내용을 포함하고 있습니다.

## Project 1. Image Clustering & Algorithm Analysis

이미지 임베딩 데이터를 활용하여 다양한 군집화 알고리즘의 성능을 비교하고, 각 알고리즘의 특성을 분석했습니다.

### 1) 주요 수행 과업

- **Custom K-Means 구현**: 라이브러리를 사용하지 않고 중심점 초기화, 할당, 업데이트 과정을 직접 구현하여 알고리즘의 동작 원리 이해

- **다양한 군집화 알고리즘 비교**: DBSCAN, HDBSCAN 모델을 적용하여 데이터 분포에 따른 최적의 알고리즘 탐색

- **차원 축소 및 시각화**: 고차원 이미지 임베딩 데이터를 PCA와 t-SNE를 통해 2차원으로 축소하여 클러스터링 결과 시각화

- **정량적 성능 평가**: 다음과 같은 지표를 활용하여 군집화 품질 검증
  - **SSE (Sum of Squared Errors)**: 군집 내 응집도 측정

  - **Silhouette Score**: 군집 간 분리도 및 군집 내 밀집도 평가

  - **Dunn Index**: 군집 간 최소 거리와 군집 내 최대 거리의 비율 측정

$$D = \frac{\min_{1 \le i < j \le K} d(C_i, C_j)}{\max_{1 \le k \le K} \Delta C_k}$$

---

## Project 2. Hybrid Stock Price Prediction

S&P 500 IT 섹터 종목을 대상으로, 시계열 데이터와 재무 지표를 결합한 하이브리드 딥러닝 모델을 구축했습니다.

### 1) 모델 아키텍처 (Hybrid Architecture)

단순한 가격 정보뿐만 아니라 기업의 펀더멘털 데이터를 함께 학습하는 구조를 설계했습니다.

- **Sequence Branch**: RNN, LSTM, Transformer 모델을 각각 사용하여 OHLCV(시가, 고가, 저가, 종가, 거래량) 시계열 패턴 학습

- **Static Branch**: PER, PBR, ROE와 같은 기업 고유의 정적 데이터를 MLP(Multi-Layer Perceptron)로 처리

- **Feature Fusion**: 시계열 특징점과 정적 특징점을 결합(Concatenate)하여 최종 주가 예측

### 2) 기술적 특징

- **데이터 파이프라인**: `yfinance` API를 활용한 실시간 데이터 수집 및 `MinMaxScaler`를 이용한 정규화

- **성능 비교 분석**: 동일한 데이터셋에서 **RNN**, **LSTM**, **Transformer**의 학습 손실(MSE Loss) 및 예측 정확도 비교

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

---

## 🛠 Tech Stack

- **Language**: Python
- **Data Analysis**: Pandas, NumPy, Scikit-learn, Scipy
- **Deep Learning**: PyTorch
- **Visualization**: Matplotlib, Seaborn, PIL
- **Algorithms**: K-Means, DBSCAN, HDBSCAN, RNN, LSTM, Transformer

---

## 💡 Key Results

- **Clustering**: 데이터의 밀도와 분포에 따라 K-Means보다 DBSCAN/HDBSCAN이 노이즈 처리에 더 효과적임을 확인했습니다.

- **Stock Prediction**: Transformer 모델이 장기 의존성(Long-term dependency)을 파악하는 데 있어 RNN 기반 모델보다 낮은 최종 손실 값을 기록하는 경향을 보였습니다.
