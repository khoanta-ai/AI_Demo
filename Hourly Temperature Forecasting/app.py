import torch
import torch.nn as nn
import streamlit as st
import numpy as np
import pandas as pd

class WeatherForecastor(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_layers, dropout_prob):
        super(WeatherForecastor, self).__init__()
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, hn = self.rnn(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Load the model
embedding_dim = 1
hidden_size = 8
n_layers = 3
dropout_prob = 0.2
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

model = WeatherForecastor(
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    n_layers=n_layers,
    dropout_prob=dropout_prob
).to(device)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Streamlit app
st.title('Hourly Temperature Forecasting')
test_data = pd.read_csv('test_data.csv')

# User input for index
index = st.number_input("Enter the index of the data to display:", min_value=0, max_value=len(test_data)-1, value=0)

# Display test data at the specified index
st.markdown("## Test Data")
st.write(test_data.iloc[index])

# User input for temperature data
if 'temp_data' not in st.session_state:
    st.session_state.temp_data = [0.0] * 6  # Initialize with default values

if st.button('Apply'):
    for i in range(6):
        temp = st.number_input(f"Temperature at Hour {i+1}:", min_value=-100.0, max_value=100.0, value=test_data.iloc[index]['Temp at Hour ' + str(i+1)], key=f"temp_input_{i}")
        st.session_state.temp_data[i] = temp  # Store input in session state
else:
    for i in range(6):
        temp = st.number_input(f"Temperature at Hour {i+1}:", min_value=-100.0, max_value=100.0, value=st.session_state.temp_data[i], key=f"temp_input_{i}")
        st.session_state.temp_data[i] = temp  # Store input in session state

# Convert user input to tensor with the required shape
temp_tensor = torch.FloatTensor(st.session_state.temp_data).unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions

# Predict button
if st.button('Predict'):
    # Make prediction
    with torch.no_grad():
        prediction = model(temp_tensor)
        predicted_temp = prediction.item()

    st.markdown("## Predicted Temperature for the Next Hour:")
    st.write(f"Predicted Temperature: {predicted_temp:.2f}°C")