# USD_CAD_exchange_rate_predictor

In this project, I attempted to predict CAD to USD exchange rates in 30 days given previous exchange rates. This project is mainly for me to practice analyzing time series data and applying machine learning methods on those data. Through this project, I realized how difficult it is to make prediction in practice and feature engineering has great potentials in improving predictions. If time allows, I would like to try more machine learning methods and try more feature engineering methods.

The CAD to USD exchange rates data used in this project are obtained using the [Alpha Vantage API](https://www.alphavantage.co/documentation/). The data include daily CAD to USD exchange rates from 2018-01-01 to 2019-12-31.
Only daily close data are used for this project. Data were imputated with last observation carried over imputation to fill any gaps.

## Data Analysis and Report

The exploratory data analysis and complete data analysis is in the [eda
notebook](./scripts/eda.ipynb)
and the final report can be found
[here](https://github.com/flizhou/CAD_USD_exchange_rate_predictor/blob/master/doc/report.md)

## Dependencies

- Python 3.7.4 and Python packages:
    - pandas==0.25.2
    - requests==2.23.0
    - numpy==1.18.1
    - scipy==1.4.1
    - scikit-learn==0.23.1
    - matplotlib==3.2.1
    - statsmodels==0.11.1
    - Keras==2.3.1
    - tensorflow==2.2.0
    - pmdarima==1.6.1
    - tqdm==4.42.1
    - lightgbm==4.42.1
    - json5==0.9.0
- R 3.6.1 and R packages:
    - knitr==1.27.2
    - tidyverse==1.3.0
