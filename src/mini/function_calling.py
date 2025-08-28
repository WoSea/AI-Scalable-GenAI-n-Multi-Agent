import json
import requests
import os
import webbrowser
from dotenv import load_dotenv
import ollama

# Load env
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ---- Fake booking API ----
def booking_api(checkin: str, checkout: str, people: int):
    return {
        "hotel": "Grand Riverside",
        "checkin": checkin,
        "checkout": checkout,
        "people": people,
        "price": 120 * people,
        "currency": "USD",
    }

# ---- Real weather API ----
def get_weather(city: str):
    # Get Geo coords from another api https://openweathermap.org/find?q=Ho+CHI+MINH
    lat = 10.8333
    lon = 106.6667
    # Call API
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&lang=vi"
    r = requests.get(url)
    data = r.json()

    if r.status_code != 200 or "main" not in data:
        return {"error": "Can not get weather data"}
    
    weather = {
        "city": city,
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"],
        "link": "https://www.accuweather.com/vi/vn/ho-chi-minh-city/353981/hourly-weather-forecast/353981"
    }
    return weather

# Function schema
functions = [
    {
        "name": "booking_api",
        "description": "Get hotel booking information",
        "parameters": {
            "type": "object",
            "properties": {
                "checkin": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                "checkout": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                "people": {"type": "integer", "description": "Number of people"},
            },
            "required": ["checkin", "checkout", "people"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get real-time weather information for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name (e.g. 'Ho Chi Minh City')"},
            },
            "required": ["city"],
        },
    }
]

# Chatbot
def chatbot():
    print("Assistant (type 'quit' to exit)")
    context = []
    while True:
        user_input = input("You: ")
        # Tôi muốn đặt phòng từ ngày 29/8 đến 30/8 cho 2 người # I want to book a room from August 29 to August 30 for 2 people
        # Thời tiết ở thành phố Hồ Chí Minh hôm nay thế nào? # How is the weather in Ho Chi Minh City today?
        if user_input.lower() == "quit":
            break

        # Ollama
        response = ollama.chat(
            model="mistral",
            messages=context + [{"role": "user", "content": user_input}],
            options={"functions": functions},
        )

        msg = response["message"]

        if "function_call" in msg:
            fn = msg["function_call"]["name"]
            args = json.loads(msg["function_call"]["arguments"])
            print(f"[DEBUG] Calling function {fn} with args {args}")

            if fn == "booking_api":
                result = booking_api(**args)
            elif fn == "get_weather":
                result = get_weather(**args)
            else:
                result = {"error": "Function not implemented"}

            # Add function result to context
            context.append({"role": "user", "content": user_input})
            context.append(msg)  # function call
            context.append({"role": "function", "name": fn, "content": json.dumps(result)})

            # Model continues to respond based on function result
            final_response = ollama.chat(model="mistral", messages=context)
            final_text = final_response["message"]["content"]
            print("Assistant:", final_text)
            context.append(final_response["message"])

            # Send data to API
            try:
                save_url = "http://localhost/api/v1/data/save"
                payload = {"query": user_input, "response": final_text}
                r = requests.post(save_url, json=payload)
                print(f"[ACTION] Posted to {save_url}, status={r.status_code}")
            except Exception as e:
                print("[ERROR] Failed to post data:", e)

            # open browser & post API
            if fn == "get_weather":
                # Open web browser to AccuWeather
                link = result.get("link")
                if link:
                    print(f"[ACTION] Opening browser: {link}")
                    webbrowser.open(link)
        else:
            print("Assistant:", msg["content"])
            context.append({"role": "user", "content": user_input})
            context.append(msg)

if __name__ == "__main__":
    chatbot()