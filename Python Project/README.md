# Bitcoin Price Prediction Application

This project is a Bitcoin price prediction application that utilizes a recurrent neural network (RNN) to forecast future prices based on historical data. The application fetches real-time data and visualizes predictions using Plotly and Streamlit.

## Project Structure

```
Python Project
├── Code
│   ├── rnnpricepredict.py  # Main code for the application
│   └── requirements.txt     # List of dependencies
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Navigate to the Code directory**:
   ```bash
   cd Code
   ```

3. **Install the required dependencies**:
   You can install the required libraries using pip. Make sure you have Python and pip installed on your machine.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the Bitcoin price prediction application, execute the following command in your terminal:

```bash
python rnnpricepredict.py
```

The application will start, and you can interact with the Streamlit interface to select different intervals for predictions.

## Dependencies

The following libraries are required to run this project:

- pandas
- yfinance
- keras
- scikit-learn
- numpy
- plotly
- streamlit
- pytz

Make sure to install these libraries as mentioned in the setup instructions.