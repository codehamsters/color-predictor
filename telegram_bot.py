import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import requests
import os
from telegram import Bot
from telegram.ext import CommandHandler, Updater
import asyncio


TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = Bot(token=TOKEN)

class ColorPredictor:
    def __init__(self):
        self.model = Sequential([
            Dense(16, activation='relu', input_shape=(1,)),
            Dense(32, activation='relu'),
            Dropout(0.2),  # Adding dropout for regularization
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.X = np.array([])  # Features
        self.y = np.array([])  # Labels (0 for red, 1 for green)

    def train(self, color, is_correct):
        if 'red' in color:
            label = 0
        elif 'green' in color:
            label = 1
        else:
            raise ValueError("Invalid color")

        self.X = np.append(self.X, 1)  # Bias term
        self.y = np.append(self.y, label)
        self.model.fit(self.X, self.y, epochs=20, verbose=0, batch_size=8)  # Adjusting epochs and batch size

    def predict_color(self):
        if len(self.X) == 0:  # Check if the model is trained
            return ['red',0]  # Default prediction if model is not trained

        prob_red = self.model.predict(np.array([[1]]))[0, 0]  # Predict probability of red
        if prob_red >= 0.5:
            return ['red', prob_red]
        else:
            return ['green', 1 - prob_red]

def fetch_initial_data():
    # Fetch initial data from the API
    url = "https://lucky66-api.glitch.me/INR/game/record"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch initial data from the API.")

def fetch_color_for_period():
    # Fetch color for a specific period from the API
    url = "https://lucky66-api.glitch.me/INR/game/record/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data[0]
    else:
        raise Exception(f"Failed to fetch data from the API.")

async def main():
    predictor = ColorPredictor()

    # Fetch initial data from the API
    initial_data = fetch_initial_data()

    # Iterate through the initial data to train the model and make predictions
    for data in initial_data[::-1]:  # Reverse the list to start from the beginning
        color = data['color']
        # await bot.send_message(chat_id = CHAT_ID, text=color)
        predictor.train(color, 1)  # Assuming all initial predictions are correct
    await bot.send_message(chat_id=CHAT_ID, text="Initial training complete")
    prev_data = initial_data[0]
    while True:
        predicted_color = predictor.predict_color()
        await bot.send_message(chat_id=CHAT_ID, text=f"Predicted {prev_data['period'].split('S')[0] + 'S' + str(int(prev_data['period'].split('S')[1])+1)}:{predicted_color}")
        time.sleep(180)
        new_data = fetch_color_for_period()
        while new_data["period"] == prev_data["period"]:
            new_data = fetch_color_for_period()
        
        actual_color = new_data["color"]
        await bot.send_message(chat_id=CHAT_ID, text=f"Actual color for period {new_data['period']}: {actual_color}")
        is_correct = 1 if predicted_color[0] in actual_color else 0
        predictor.train(predicted_color[0], is_correct)
        prev_data=new_data

if __name__ == "__main__":
    asyncio.run(main())
