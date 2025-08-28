# pip install webbrowser-open dotenv ollama
# pip install geopy
# https://ollama.com/search?c=tools&q=llama
# ollama pull llama3.2
# ollama run llama3.2 to check model
# Create .env file with content: OPENWEATHER_API_KEY="your_api"

import json
import requests
import os
import webbrowser_open
from dotenv import load_dotenv
import ollama

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my_geopy_app")

# Load env
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Fake booking API
def booking_api(checkin: str, checkout: str, people: int):
    price = 120 * int(people)
    return {
        "hotel": "Grand Riverside HCM",
        "checkin": checkin,
        "checkout": checkout,
        "people": people,
        "price": price,
        "currency": "USD",
    }

# Real weather API
def get_weather(city: str):
    # Get Geo coords from another api https://openweathermap.org/find?q=Ho+CHI+MINH
    lat = 10.8333
    lon = 106.6667

    location = geolocator.geocode(str(city))
    if location:
        lat = location.latitude
        lon = location.longitude
        print(f"Geolocation for {city}: Latitude={lat}, Longitude={lon}")
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
        "link": "https://www.accuweather.com/en/world-weather"  # Replace this link with your own api/web
    }
    print(f"\nweather: {weather}")
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
            }
        }
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
            }
        }
}

# Chatbot
SYSTEM_PROMPT = """You are a helpful AI assistant.
- Always use results from tools when available.
- If user booked a hotel, confirm booking details (hotel name, checkin, checkout, people, price).
- If user asked about weather, summarize weather clearly (temperature °C, humidity %, description).
- If both booking and weather are available, combine results naturally in one answer.
- If weather info is provided, remind the user they can check more at the provided link.
- Be concise and friendly.
"""
def chatbot():
    print("Assistant (type 'quit' to exit)")
    context = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # Ollama
        response = ollama.chat(
            model="llama3.2",
            messages=context + [{"role": "user", "content": user_input}],
            tools=[booking_api_function, get_weather_function],
        )
        print(f"\nresponse = ollama.chat: {response}") 
        msg = response["message"]
        print(f"\n\n MSG: {msg}")

        print(f"\n\n msg.tool_calls: {msg.tool_calls}")
        if msg.tool_calls:
            tool_results = []
            have_get_weather_function = False
            for tool in msg.tool_calls:
                if tool.function.name == 'booking_api':
                    result = booking_api(**tool.function.arguments)
                    print(f"\nresult 1: {result}")
                elif tool.function.name == 'get_weather':
                    have_get_weather_function = True
                    result = get_weather(**tool.function.arguments)
                    print(f"\nresult 2: {result}")
                else:
                    result = {"error": "Function not implemented"}
                    print(f"\nresult 3: {result}")

                tool_results.append({
                    "role": "tool",
                    "name": tool.function.name,
                    "content": json.dumps(result)
                })
            print(f"\n tool_results: {tool_results}")

            # Add function result to context
            context.append({"role": "user", "content": user_input})
            context.append(msg)  # function call
            context.extend(tool_results)

            print(f"\n\n[DEBUG] Context: {context}")

            # Model continues to respond based on function result
            final_response = ollama.chat(model="llama3.2", messages=context)
            final_text = final_response["message"]["content"]
            print(f"\n Assistant: {final_text}")
            context.append(final_response["message"])

            # Send data to API
            try:
                save_url = "http://localhost/api/v1/data/save"
                payload = {"query": user_input, "response": final_text}
                r = requests.post(save_url, json=payload)
                print(f"[ACTION] Posted to {save_url}, status={r.status_code}")
            except Exception as e:
                print("[ERROR] Failed to post data:", e)

            # open browser if weather link
            if have_get_weather_function:
                for tr in tool_results:
                    if tr["name"] == "get_weather":
                        link = json.loads(tr["content"]).get("link")
                        print(f"\nlink: {link}")
                        if link:
                            print(f"\n[ACTION] Opening browser: {link}")
                            webbrowser_open.open(link)

        else:
            print(f"\n Assistant: {msg['content']}")
            context.append({"role": "user", "content": user_input})
            context.append(msg)

if __name__ == "__main__":
    chatbot()

# I want to book a room from August 29 to August 30 for 4 people at Ho Chi Minh City. How is the weather in Ho Chi Minh City these days?

''' 
(base) root@gpu3:~/duy/AI-Scalable-GenAI-n-Multi-Agent/src/mini# python function_calling.py
Assistant (type 'quit' to exit)
You: I want to book a room from August 29 to August 30 for 4 people at Ho Chi Minh City. How is the weather in Ho Chi Minh City these days?

response = ollama.chat: model='llama3.2' created_at='2025-08-28T19:20:55.509496186Z' done=True done_reason='stop' total_duration=2472020337 load_duration=1901299769 prompt_eval_count=391 prompt_eval_duration=157638695 eval_count=48 eval_duration=411277362 message=Message(role='assistant', content='', thinking=None, images=None, tool_name=None, tool_calls=[ToolCall(function=Function(name='booking_api', arguments={'checkin': '2023-08-29', 'checkout': '2023-08-30', 'people': '4'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'Ho Chi Minh City'}))])


 MSG: role='assistant' content='' thinking=None images=None tool_name=None tool_calls=[ToolCall(function=Function(name='booking_api', arguments={'checkin': '2023-08-29', 'checkout': '2023-08-30', 'people': '4'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'Ho Chi Minh City'}))]


 msg.tool_calls: [ToolCall(function=Function(name='booking_api', arguments={'checkin': '2023-08-29', 'checkout': '2023-08-30', 'people': '4'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'Ho Chi Minh City'}))]

result 1: {'hotel': 'Grand Riverside', 'checkin': '2023-08-29', 'checkout': '2023-08-30', 'people': '4', 'price': 480, 'currency': 'USD'}
Geolocation for Ho Chi Minh City: Latitude=10.7755254, Longitude=106.7021047
Weather API response: {'coord': {'lon': 106.7023, 'lat': 10.781}, 'weather': [{'id': 804, 'main': 'Clouds', 'description': 'mây đen u ám', 'icon': '04n'}], 'base': 'stations', 'main': {'temp': 25.4, 'feels_like': 26.29, 'temp_min': 25.4, 'temp_max': 25.4, 'pressure': 1007, 'humidity': 88, 'sea_level': 1007, 'grnd_level': 1006}, 'visibility': 10000, 'wind': {'speed': 2.87, 'deg': 268, 'gust': 9.58}, 'clouds': {'all': 100}, 'dt': 1756408834, 'sys': {'country': 'VN', 'sunrise': 1756421026, 'sunset': 1756465505}, 'timezone': 25200, 'id': 1566083, 'name': 'Thành phố Hồ Chí Minh', 'cod': 200}

weather: {'city': 'Ho Chi Minh City', 'temperature': 25.4, 'humidity': 88, 'description': 'mây đen u ám', 'link': 'https://www.accuweather.com/en/world-weather'}

result 2: {'city': 'Ho Chi Minh City', 'temperature': 25.4, 'humidity': 88, 'description': 'mây đen u ám', 'link': 'https://www.accuweather.com/en/world-weather'}

 tool_results: [{'role': 'tool', 'name': 'booking_api', 'content': '{"hotel": "Grand Riverside", "checkin": "2023-08-29", "checkout": "2023-08-30", "people": "4", "price": 480, "currency": "USD"}'}, {'role': 'tool', 'name': 'get_weather', 'content': '{"city": "Ho Chi Minh City", "temperature": 25.4, "humidity": 88, "description": "m\\u00e2y \\u0111en u \\u00e1m", "link": "https://www.accuweather.com/en/world-weather"}'}]


[DEBUG] Context: [{'role': 'system', 'content': 'You are a helpful AI assistant. \n- Always use results from tools when available. \n- If user booked a hotel, confirm booking details (hotel name, checkin, checkout, people, price). \n- If user asked about weather, summarize weather clearly (temperature °C, humidity %, description). \n- If both booking and weather are available, combine results naturally in one answer. \n- If weather info is provided, remind the user they can check more at the provided link.\n- Be concise and friendly.\n'}, {'role': 'user', 'content': 'I want to book a room from August 29 to August 30 for 4 people at Ho Chi Minh City. How is the weather in Ho Chi Minh City these days?'}, Message(role='assistant', content='', thinking=None, images=None, tool_name=None, tool_calls=[ToolCall(function=Function(name='booking_api', arguments={'checkin': '2023-08-29', 'checkout': '2023-08-30', 'people': '4'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'Ho Chi Minh City'}))]), {'role': 'tool', 'name': 'booking_api', 'content': '{"hotel": "Grand Riverside", "checkin": "2023-08-29", "checkout": "2023-08-30", "people": "4", "price": 480, "currency": "USD"}'}, {'role': 'tool', 'name': 'get_weather', 'content': '{"city": "Ho Chi Minh City", "temperature": 25.4, "humidity": 88, "description": "m\\u00e2y \\u0111en u \\u00e1m", "link": "https://www.accuweather.com/en/world-weather"}'}]



Assistant: You have successfully booked a room at the Grand Riverside hotel in Ho Chi Minh City from August 29 to August 30 for 4 people, with a total price of $480.

As for the weather, Ho Chi Minh City is currently experiencing warm and humid conditions. The temperature is around 25.4°C (77.7°F), with high humidity at 88%. You can check more weather updates on AccuWeather's website: https://www.accuweather.com/en/world-weather

Enjoy your stay in Ho Chi Minh City!
[ACTION] Posted to http://localhost/api/v1/data/save, status=200

link: https://www.accuweather.com/en/world-weather

[ACTION] Opening browser: https://www.accuweather.com/en/world-weather
You: 
'''

# I want to book a room from August 30 to September 2 for 4 people at New York City. How is the weather in New York City these days?
'''
(base) root@gpu3:~/duy/AI-Scalable-GenAI-n-Multi-Agent/src/mini# python function_calling.py
Assistant (type 'quit' to exit)
You: I want to book a room from August 30 to September 2 for 4 people at New York City. How is the weather in New York City these days?

response = ollama.chat: model='llama3.2' created_at='2025-08-28T19:23:40.602994284Z' done=True done_reason='stop' total_duration=830469108 load_duration=94024187 prompt_eval_count=389 prompt_eval_duration=21747897 eval_count=79 eval_duration=712884292 message=Message(role='assistant', content='', thinking=None, images=None, tool_name=None, tool_calls=[ToolCall(function=Function(name='booking_api', arguments={'checkin': '2023-08-30', 'checkout': '2023-09-02', 'people': '4'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'New York City'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'New York City'}))])


 MSG: role='assistant' content='' thinking=None images=None tool_name=None tool_calls=[ToolCall(function=Function(name='booking_api', arguments={'checkin': '2023-08-30', 'checkout': '2023-09-02', 'people': '4'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'New York City'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'New York City'}))]


 msg.tool_calls: [ToolCall(function=Function(name='booking_api', arguments={'checkin': '2023-08-30', 'checkout': '2023-09-02', 'people': '4'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'New York City'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'New York City'}))]

result 1: {'hotel': 'Grand Riverside', 'checkin': '2023-08-30', 'checkout': '2023-09-02', 'people': '4', 'price': 480, 'currency': 'USD'}
Geolocation for New York City: Latitude=40.7127281, Longitude=-74.0060152
Weather API response: {'coord': {'lon': -74.006, 'lat': 40.7127}, 'weather': [{'id': 804, 'main': 'Clouds', 'description': 'mây đen u ám', 'icon': '04d'}], 'base': 'stations', 'main': {'temp': 25.08, 'feels_like': 24.77, 'temp_min': 25.08, 'temp_max': 25.08, 'pressure': 1018, 'humidity': 43, 'sea_level': 1018, 'grnd_level': 1017}, 'visibility': 10000, 'wind': {'speed': 6.44, 'deg': 163, 'gust': 6.58}, 'clouds': {'all': 100}, 'dt': 1756409023, 'sys': {'country': 'US', 'sunrise': 1756376371, 'sunset': 1756424118}, 'timezone': -14400, 'id': 5128581, 'name': 'Thành phố New York', 'cod': 200}

weather: {'city': 'New York City', 'temperature': 25.08, 'humidity': 43, 'description': 'mây đen u ám', 'link': 'https://www.accuweather.com/en/world-weather'}

result 2: {'city': 'New York City', 'temperature': 25.08, 'humidity': 43, 'description': 'mây đen u ám', 'link': 'https://www.accuweather.com/en/world-weather'}
Geolocation for New York City: Latitude=40.7127281, Longitude=-74.0060152
Weather API response: {'coord': {'lon': -74.006, 'lat': 40.7127}, 'weather': [{'id': 804, 'main': 'Clouds', 'description': 'mây đen u ám', 'icon': '04d'}], 'base': 'stations', 'main': {'temp': 25.08, 'feels_like': 24.77, 'temp_min': 25.08, 'temp_max': 25.08, 'pressure': 1018, 'humidity': 43, 'sea_level': 1018, 'grnd_level': 1017}, 'visibility': 10000, 'wind': {'speed': 6.44, 'deg': 163, 'gust': 6.58}, 'clouds': {'all': 100}, 'dt': 1756409023, 'sys': {'country': 'US', 'sunrise': 1756376371, 'sunset': 1756424118}, 'timezone': -14400, 'id': 5128581, 'name': 'Thành phố New York', 'cod': 200}

weather: {'city': 'New York City', 'temperature': 25.08, 'humidity': 43, 'description': 'mây đen u ám', 'link': 'https://www.accuweather.com/en/world-weather'}

result 2: {'city': 'New York City', 'temperature': 25.08, 'humidity': 43, 'description': 'mây đen u ám', 'link': 'https://www.accuweather.com/en/world-weather'}

 tool_results: [{'role': 'tool', 'name': 'booking_api', 'content': '{"hotel": "Grand Riverside", "checkin": "2023-08-30", "checkout": "2023-09-02", "people": "4", "price": 480, "currency": "USD"}'}, {'role': 'tool', 'name': 'get_weather', 'content': '{"city": "New York City", "temperature": 25.08, "humidity": 43, "description": "m\\u00e2y \\u0111en u \\u00e1m", "link": "https://www.accuweather.com/en/world-weather"}'}, {'role': 'tool', 'name': 'get_weather', 'content': '{"city": "New York City", "temperature": 25.08, "humidity": 43, "description": "m\\u00e2y \\u0111en u \\u00e1m", "link": "https://www.accuweather.com/en/world-weather"}'}]


[DEBUG] Context: [{'role': 'system', 'content': 'You are a helpful AI assistant. \n- Always use results from tools when available. \n- If user booked a hotel, confirm booking details (hotel name, checkin, checkout, people, price). \n- If user asked about weather, summarize weather clearly (temperature °C, humidity %, description). \n- If both booking and weather are available, combine results naturally in one answer. \n- If weather info is provided, remind the user they can check more at the provided link.\n- Be concise and friendly.\n'}, {'role': 'user', 'content': 'I want to book a room from August 30 to September 2 for 4 people at New York City. How is the weather in New York City these days?'}, Message(role='assistant', content='', thinking=None, images=None, tool_name=None, tool_calls=[ToolCall(function=Function(name='booking_api', arguments={'checkin': '2023-08-30', 'checkout': '2023-09-02', 'people': '4'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'New York City'})), ToolCall(function=Function(name='get_weather', arguments={'city': 'New York City'}))]), {'role': 'tool', 'name': 'booking_api', 'content': '{"hotel": "Grand Riverside", "checkin": "2023-08-30", "checkout": "2023-09-02", "people": "4", "price": 480, "currency": "USD"}'}, {'role': 'tool', 'name': 'get_weather', 'content': '{"city": "New York City", "temperature": 25.08, "humidity": 43, "description": "m\\u00e2y \\u0111en u \\u00e1m", "link": "https://www.accuweather.com/en/world-weather"}'}, {'role': 'tool', 'name': 'get_weather', 'content': '{"city": "New York City", "temperature": 25.08, "humidity": 43, "description": "m\\u00e2y \\u0111en u \\u00e1m", "link": "https://www.accuweather.com/en/world-weather"}'}]


Assistant: You have successfully booked a room at the Grand Riverside hotel in New York City from August 30 to September 2 for 4 people, with a price of $480.

As for the weather, it's currently quite pleasant in New York City. The temperature is around 25°C (77°F), with a relative humidity of 43%. There's a gentle rain shower expected today.

You can check the latest weather forecast and more details on the AccuWeather website: https://www.accuweather.com/en/world-weather
[ACTION] Posted to http://localhost/api/v1/data/save, status=200

link: https://www.accuweather.com/en/world-weather

[ACTION] Opening browser: https://www.accuweather.com/en/world-weather

link: https://www.accuweather.com/en/world-weather

[ACTION] Opening browser: https://www.accuweather.com/en/world-weather
You: 
'''



