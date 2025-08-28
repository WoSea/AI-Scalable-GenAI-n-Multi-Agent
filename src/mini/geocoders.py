# pip install geopy
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my_geopy_app")
city_name = "Paris, France"

location = geolocator.geocode(city_name)

if location:
    print(f"The coordinates for {city_name} are:")
    print(f"Latitude: {location.latitude}")
    print(f"Longitude: {location.longitude}")
    print(f"Full Address: {location.address}")
else:
    print(f"Could not find geolocation for {city_name}.")

'''
The coordinates for Paris, France are:
Latitude: 48.8534951
Longitude: 2.3483915
Full Address: Paris, Île-de-France, France métropolitaine, France
'''