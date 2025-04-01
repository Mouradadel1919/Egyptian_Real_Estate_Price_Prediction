# Egyptian Real Estate Price Prediction


A machine learning pipeline for predicting real estate prices in Egypt, from data collection to production deployment.

## 📖 Overview
This project scraped over 300,000 property listings using Selenium in Python, followed by comprehensive data preprocessing, feature engineering, and machine learning modeling. The final pipeline handles raw data through inflation-adjusted price conversion and delivers predictions via a Dockerized Flask API. The complete workflow includes:
- Web scraping (Selenium)
- Data preprocessing & EDA
- Inflation adjustment using Central Bank of Egypt data
- XGBoost regression modeling
- Flask API deployment
- Docker containerization

## 🕷️ Selenium Web Scraping
Collected 300,000+ property listings with detailed features:
- Location
- Price per square meter
- Property specifications (rooms, bathrooms, floor)
- Quality metrics (finish type, view type)
- Temporal features (construction year)
- Financial details (payment type)


## 📊 Data Preprocessing & EDA
### Key Challenges & Solutions:
- **Inflation Adjustment**: Integrated [Central Bank of Egypt](https://www.cbe.org.eg/en) inflation rates to convert prices to 2024 equivalents
- Feature Engineering:
  - Time-based feature extraction
  - Categorical encoding
- Visual Analysis:
  - Location Analysis
  - Price distribution mapping
  - Feature correlation matrices

## 🤖 Modeling
### Model Development:
- Baseline: Linear Regression (rejected due to violated assumptions)
- Final Model: XGBoost Regressor with hyperparameter tuning

## 📈 Evaluation
| Metric            | Value   |
|-------------------|---------|
| Mean Squared Error| 0.0182  |
| R² Score          | 0.96    |


## 🚀 Deployment


### Docker Deployment:

Docker Hub:- https://hub.docker.com/repository/docker/mouradadel313/real_estate/tags

```bash
docker pull mouradadel313/real_estate:latest
docker run -d -p 5000:5000 mouradadel313/real_estate:latest

