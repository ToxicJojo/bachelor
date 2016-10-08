import json
import sys
import requests
import os

IMG_FILE_DIR = "data/img/"

def downloadCardImages(pathToSetFile): 
    # Load the json file containig all the data for the specific set.
    setFile = open(pathToSetFile)
    setData = json.load(setFile)

    # Create the folder that will contain all the card images.
    setCode = setData["code"]
    os.makedirs(IMG_FILE_DIR + setCode, exist_ok=True)

    cards = setData["cards"]

    for card in cards:
        # The multiverseid is a unique identifier for a card.
        cardID = str(card["multiverseid"])
        cardName = card["name"]

        print("Downloading: " + cardName + " (" + cardID + ")")
        # Download the card image and write it to a file $cardName.png.
        r = requests.get("http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=" + cardID + "&type=card")
        with open(IMG_FILE_DIR + setCode + "/" + cardName + ".png", 'wb') as fd:
            for chunk in r.iter_content(200):
                fd.write(chunk)
