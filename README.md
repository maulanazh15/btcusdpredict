# Bitcoin Price Prediction Application

This project is a Bitcoin price prediction application that utilizes a recurrent neural network (RNN) to forecast future prices based on historical data. The application fetches real-time data and visualizes predictions using Plotly and Streamlit.

## Project Structure

```
Bitcoin Price Prediction
├── README.md                # Project documentation
├── requirements.txt         # List of dependencies
├── rnnbtc1d.keras           # Pre-trained model for 1-day interval
├── rnnbtc1h.keras           # Pre-trained model for 1-hour interval
├── rnnbtc5min.keras         # Pre-trained model for 5-minute interval
├── rnnforecastbtc1d.ipynb   # Jupyter Notebook for 1-day interval forecasting
├── rnnforecastbtc1h.ipynb   # Jupyter Notebook for 1-hour interval forecasting
├── rnnforecastbtc5min.ipynb # Jupyter Notebook for 5-minute interval forecasting
└── rnnpricepredict.py       # Main code for the application
```

## Setup Instructions

1. **Clone the repository** (if applicable):
   ```bash
   git clone https://github.com/maulanazh15/btcusdpredict.git
   cd btcusdpredict
   ```

2. **Navigate to the project directory**:
   ```bash
   cd "Bitcoin Price Prediction"
   ```

3. **Install the required dependencies**:
   You can install the required libraries using pip. Make sure you have Python and pip installed on your machine.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the Bitcoin price prediction application, execute the following command in your terminal:

```bash
streamlit run rnnpricepredict.py
```

The application will start, and you can interact with the Streamlit interface to select different intervals for predictions.

## Pre-trained Models

The application uses the following pre-trained models for predictions:
- **`rnnbtc1d.keras`**: Model for 1-day interval predictions.
- **`rnnbtc1h.keras`**: Model for 1-hour interval predictions.
- **`rnnbtc5min.keras`**: Model for 5-minute interval predictions.

## Jupyter Notebooks

The project includes Jupyter Notebooks for detailed forecasting and experimentation:
- **`rnnforecastbtc1d.ipynb`**: Forecasting for 1-day intervals.
- **`rnnforecastbtc1h.ipynb`**: Forecasting for 1-hour intervals.
- **`rnnforecastbtc5min.ipynb`**: Forecasting for 5-minute intervals.

## Dependencies

The following libraries are required to run this project:

- pandas
- yfinance
- keras
- scikit-learn
- numpy
- plotly
- streamlit

Make sure to install these libraries as mentioned in the setup instructions.