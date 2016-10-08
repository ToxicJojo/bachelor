import requests
import os
import sys

def downloadSetInfo(setCode):
    # Create the data folder if it
    os.makedirs("data", exist_ok=True)


    print("Donwloading: https://mtgjson.com/json/" + setCode + ".json")
    r = requests.get("https://mtgjson.com/json/" + setCode + ".json")

    with open("data/" + setCode + ".json", 'wb') as fd:
        for chunk in r.iter_content(200):
            fd.write(chunk)