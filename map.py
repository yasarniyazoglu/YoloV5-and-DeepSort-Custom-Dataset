import requests, json
from geopy.geocoders import Nominatim

def current_location():
    here_req = requests.get("http://www.geoplugin.net/json.gp")

    if (here_req.status_code != 200):
        print("현재좌표를 불러올 수 없음")
    else:
        location = json.loads(here_req.text)
        crd = {"lat": str(location["geoplugin_latitude"]), "lng": str(location["geoplugin_longitude"])}

    return crd

def geocoding_reverse(lat_lng_str): 
    geolocoder = Nominatim(user_agent = 'South Korea', timeout=None)
    address = geolocoder.reverse(lat_lng_str)

    return address

def address():
    crd = current_location()
    #print(crd)
    address = geocoding_reverse(crd['lat'] + ", " + crd['lng'])
    #address = geocoding_reverse('35.23340816340583, 129.08200272674122')
    #print(address)
    address = str(address).split(", ")
    #print(address)
    answer = ""
    for i in range(len(address)-3,-1,-1):
        answer += address[i] + " "
    return answer


