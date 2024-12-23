# Energy Consumption Prediction and Appliance Identification

Energy supply and environmental mitigation are among the major challenges facing the planet today. Consequently, priority policies must be implemented to meet the growing demand for energy while simultaneously reducing the ecological footprint. In this context, strategies promoting rational and intelligent energy consumption, efforts to improve the efficiency of current energy systems, and the integration of innovative technologies are essential to find feasible and reliable solutions.

Our project focuses on the identification and prediction of different electrical loads in a residential house. Our reinforcement learning model, based on a Long Short-Term Memory (LSTM) recurrent neural network, is able to predict new observations from previous ones by performing a deep learning process on the collected data. This model is particularly effective for complex problems characterized by non-linearity and a non-parametric form.

The proposed method can be used for modeling and forecasting time series, which is especially suitable for the problem of disaggregating the energy consumption curve from a smart meter. This approach helps optimize the accuracy of time series forecasts, enabling better energy management and prediction.



## Proposed Model

![Proposed Model](images/modelePNG.PNG
)


## Keywords
- NILM (Non-Intrusive Load Monitoring)
- Machine Learning
- Smart Monitoring
- Internet of Things (IoT)
- Home Automation
- Energy Consumption
- Python

## Objective
The goal of this project is to enhance energy efficiency in residential areas by identifying and predicting the energy consumption patterns of individual household appliances. By leveraging advanced machine learning techniques, we aim to develop a reliable model for estimating energy usage and optimizing energy management.


## Description of the Learning Database for NILM

REDD (Reference Energy Disaggregation Dataset) is a dataset built for energy disaggregation, using measurements from six different houses. The data is recorded at a frequency of approximately 1Hz, although in some cases, measurements are made every 3 seconds. 

The dataset contains energy consumption data for each house as a whole, as well as for individual circuits, labeled by the primary type of appliance on each circuit. REDD includes both low-frequency power readings and high-frequency voltage data, the latter of which can also be used for disaggregation. However, since our algorithm focuses solely on domestic electricity consumption, we will ignore the high-frequency data.

The `lowfreq/` directory contains average power readings for both the main grid and the individual circuits of each house. The data is recorded at a frequency of about once per second for the grid, and once every three seconds for the circuits.

## Technologies Used
- Python
- Reinforcement Learning
- Recurrent Neural Networks (RNN)
- Non-Intrusive Load Monitoring (NILM)

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Hamdi708/Identification-of-Electrical-Charges-by-Deep-Learning-RL-and-LSTM.git
