import os
import pickle

from cliffhanger import PROJECT_FOLDER

ASSET_PATH = os.path.join(PROJECT_FOLDER, "cliffhanger", "model")


def load_asset(filename, asset_path: str = ASSET_PATH):
    with open(os.path.join(asset_path, filename), "wb") as f:
        asset = pickle.load(f)
    return asset
