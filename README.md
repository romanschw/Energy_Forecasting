# Energy Forecasting Project
---
## Wind Power Forecasting

### Dataset Description
The SDWPF dataset spans from January 2020 to December 2021. It comprises SCADA data collected every
10 minutes from each of the 134 wind turbines in the wind farm. The 10-minute recorded data represents average
values over each interval, derived from high-frequency (1Hz) sampling by the SCADA system.

| Column | Column Name | Specification | Note |
|--------|------------|---------------|------|
| 1  | TurbID | Wind turbine ID |  |
| 2  | Day | Day of the record |  |
| 3  | Tmstamp | Created time of the record | Time zone UTC +08:00 |
| 4  | Wspd (m/s) | Wind speed at the top of the turbine | Recorded by mechanical anemometer |
| 5  | Wdir (°) | Relative wind direction, the angle between the wind direction and the turbine nacelle direction | Wind direction and nacelle direction are in degrees from true north |
| 6  | Etmp (°C) | Temperature of the surrounding environment | Measured on the outer surface of the nacelle |
| 7  | Itmp (°C) | Temperature inside the turbine nacelle |  |
| 8  | Ndir (°) | Nacelle direction, the yaw angle of the nacelle | In degrees from true north |
| 9  | Pab1 (°) | Pitch angle of blade 1 | The angle between the chord line and the rotation plane of the blade |
| 10 | Pab2 (°) | Pitch angle of blade 2 | Same as above |
| 11 | Pab3 (°) | Pitch angle of blade 3 | Same as above |
| 12 | Prtv (kW) | Reactive power |  |
| 13 | T2m (°C) | Temperature at 2m above surface (ERA5) |  |
| 14 | Sp (Pa) | Surface pressure from ERA5 |  |
| 15 | RelH | Relative humidity | Derived based on 2m dew point temperature and 2m temperature using Python package MetPy |
| 16 | Wspd_w (m/s) | Wind speed from ERA5 | At a height of 10m |
| 17 | Wdir_w (°) | Wind direction from ERA5 | At a height of 10m |
| 18 | Tp (m) | Total precipitation from ERA5 |  |
| 19 | Patv (kW) | Active power, the wind power produced by a wind turbine at a given timestamp |  |

### Current State
- Some visualizations and exploratory data analysis (EDA)
- Initial implementation of DLinear for multivariate time series forecasting

### Next Steps

#### Feature Engineering and Selection
- Merge pitch angle variables
- Test model performance by removing less important variables (correlation analysis or feature selection methods)
- Incorporate spatial variables
- Seasonal decomposition using Fourier series
- Detrending using LOESS
- Decomposition using singular spectrum analysis (SSA)

#### Hyperparameter Optimization
- Learning rate scheduling (first cycle learning rate method)
- Optimize sliding window size, batch size, epochs, and forecasting horizon

#### Models
- Experiment with averaging DLinear models over multiple horizons
- Add validation and testing for DLinear
- Implement early stopping mechanisms
- Introduce dynamic learning rate adjustments
- Aggregate turbine output and test univariate probabilistic forecasting
- Aggregate turbine output and explore ensemble/stacking models
- Implement GRU and LSTM models
- Explore hybrid modeling approaches

#### Architecture
- Improve project directory structure
- Develop an API and frontend for testing
- Transition all preprocessing steps to Polars
- Implement a basic CI/CD pipeline

---

## Electricity Price Forecasting Project

### Current State
- Retrieving data through the ENTSO-E API
- Initial exploratory data analysis (EDA)
- Some feature engineering
- Implementation of rolling linear regression from scratch

### Next Steps
- Expand feature engineering
- Apply seasonal decomposition

- Implement probabilistic forecasting
- Explore historical simulation techniques
- Utilize conformal prediction

- Combine different forecasting models
- Average predictions across different time windows

- Implement Lasso-based autoregressive (AR) models
- Develop quantile regression models

- Automate data retrieval for continuous updates and forecasting
- Create a frontend and API to present results
