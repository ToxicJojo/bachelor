import os
import getSetInfo as setInfo
import getCardImages as cardImages

# A list of the shorthand codes for all sets currently in the Standard format.
SETS_IN_STANDARD = ["BFZ", "OGW", "SOI", "EMN", "KLD"]

DATA_FOLDER = "data/"

os.makedirs(DATA_FOLDER, exist_ok=True)

for setCode in SETS_IN_STANDARD:
    # We download the json file containing the info for a set so we can get every card image using the multiverseid
    setInfo.downloadSetInfo(setCode)
    cardImages.downloadCardImages(DATA_FOLDER + setCode + ".json")