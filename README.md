### Energy forecasting project

## Wind power forecasting

Dataset description: Te SDWPF dataset spans from January 2020 to December 2021. It comprises SCADA data collected every
10minutes from each of the 134 wind turbines in the wind farm. Te 10-minute record data represents average
values over each 10-minute interval, derived from high-frequency (1Hz) sampling by the SCADA system.

| Column | Column Name | Specification | Note |
|--------|------------|---------------|------|
| 1  | TurbID | Wind turbine ID |  |
| 2  | Day | Day of the record |  |
| 3  | Tmstamp | Created time of the record | Time zone UTC +08:00 |
| 4  | Wspd (m/s) | The wind speed at the top of the turbine | Recorded by mechanical anemometer |
| 5  | Wdir (°) | Relative wind direction, which is the angle between the wind direction and the turbine nacelle direction | Wind direction and nacelle direction are in degrees from true north |
| 6  | Etmp (°C) | Temperature of the surrounding environment | Measured outer surface of the nacelle |
| 7  | Itmp (°C) | Temperature inside the turbine nacelle |  |
| 8  | Ndir (°) | Nacelle direction, the yaw angle of the nacelle | In degree from true north |
| 9  | Pab1 (°) | Pitch angle of blade 1 | The angle between the chord line and the rotation plane of the blade |
| 10 | Pab2 (°) | Pitch angle of blade 2 | Same as above |
| 11 | Pab3 (°) | Pitch angle of blade 3 | Same as above |
| 12 | Prtv (kW) | Reactive power |  |
| 13 | T2m (°C) | Temperature at 2 m above surface (ERA5) |  |
| 14 | Sp (Pa) | Surface pressure from ERA5 |  |
| 15 | RelH | Relative humidity | Derived based on 2 m dew point temperature and 2 m temperature using Python Package metpy |
| 16 | Wspd_w (m/s) | Wind speed from ERA5 | At height of 10 m |
| 17 | Wdir_w (°) | Wind direction from ERA5 | At height of 10 m |
| 18 | Tp (m) | Total precipitation from ERA5 |  |
| 19 | Patv (kW) | Active power, the wind power produced by a wind turbine at a time stamp. |  |


# Current state :
- Some visualizations
- First