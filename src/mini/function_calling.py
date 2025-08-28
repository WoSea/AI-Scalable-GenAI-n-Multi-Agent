# pip install webbrowser-open dotenv ollama
# ollama pull mistral
# ollama run mistral to check model
# Create .env file with content: OPENWEATHER_API_KEY="your_api"
# Test Query
'''
I want to book a room from August 29 to August 30 for 2 people at Ho Chi Minh City

How is the weather in Ho Chi Minh City today?
'''
import json
import requests
import os
import webbrowser_open
from dotenv import load_dotenv
import ollama

# Load env
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Fake booking API
def booking_api(checkin: str, checkout: str, people: int):
    return {
        "hotel": "Grand Riverside",
        "checkin": checkin,
        "checkout": checkout,
        "people": people,
        "price": 120 * people,
        "currency": "USD",
    }

# Real weather API
def get_weather(city: str):
    # Get Geo coords from another api https://openweathermap.org/find?q=Ho+CHI+MINH
    lat = 10.8333
    lon = 106.6667
    # Call API
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&lang=vi"
    r = requests.get(url)
    data = r.json()
    print("Weather API response:", data)
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
booking_api_function = {
        "type": "function",
        "function": {
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
}
get_weather_function = { 
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get real-time weather information for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name (e.g. 'Ho Chi Minh City')"},
                },
                "required": ["city"],
            },
        },
}

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
            tools=[booking_api, get_weather],
        )

        msg = response["message"]
        print(f"MSG: {msg}")
        '''
        Assistant (type 'quit' to exit)
        You: hi
        MSG: role='assistant' content=' Hello! How can I assist you today? Here are a few functions that I have:\n\n1. `booking_api` - Get hotel booking information. It requires the check-in date, check-out date, and number of people as parameters.\n2. `get_weather` - Get real-time weather information for a city. It requires the city name as a parameter.\n\nIf you need any help with these functions or have any other queries, feel free to ask!' thinking=None images=None tool_name=None tool_calls=None
        Assistant:  Hello! How can I assist you today? Here are a few functions that I have:

        1. `booking_api` - Get hotel booking information. It requires the check-in date, check-out date, and number of people as parameters.
        2. `get_weather` - Get real-time weather information for a city. It requires the city name as a parameter.

        If you need any help with these functions or have any other queries, feel free to ask!
        You: Tôi muốn đặt phòng từ ngày 29/8 đến 30/8 cho 2 người
        MSG: role='assistant' content='' thinking=None images=None tool_name=None tool_calls=[ToolCall(function=Function(name='booking_api', arguments={'checkin': '2023-08-29', 'checkout': '2023-08-30', 'people': 2}))]

        '''
        if msg.tool_calls:
            tool_results = []
            for tool in msg.tool_calls:
                if tool.function.name == 'booking_api_function':
                    result = booking_api(**tool.function.arguments)
                elif tool.function.name == 'get_weather_function':
                    result = get_weather(**tool.function.arguments)
                else:
                    result = {"error": "Function not implemented"}
                tool_results.append({
                    "role": "tool",
                    "name": tool.function.name,
                    "content": json.dumps(result)
                })

                # Add function result to context
                context.append({"role": "user", "content": user_input})
                context.append(msg)  # function call
                context.extend(tool_results)

                # Model continues to respond based on function result
                final_response = ollama.chat(model="mistral", messages=context)
                final_text = final_response["message"]["content"]
                print("Assistant:", final_text)
                context.append(final_response["message"])

                # Send data to API
                # try:
                #     save_url = "http://localhost/api/v1/data/save"
                #     payload = {"query": user_input, "response": final_text}
                #     r = requests.post(save_url, json=payload)
                #     print(f"[ACTION] Posted to {save_url}, status={r.status_code}")
                # except Exception as e:
                #     print("[ERROR] Failed to post data:", e)

                # open browser & post API
                if tool.function.name == "get_weather_function":
                    # Open web browser to AccuWeather
                    link = result.get("link")
                    if link:
                        print(f"[ACTION] Opening browser: {link}")
                        # webbrowser_open.open(link)
        else:
            print("Assistant:", msg["content"])
            context.append({"role": "user", "content": user_input})
            context.append(msg)

if __name__ == "__main__":
    chatbot()