import os
import json

def print_models_info():
    thisdir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(thisdir, "modelpkg_metadata.json")) as f:
        metadata = json.load(f)

    for modelpkg, pkginfo in metadata.items():
        print(f"Model Package: '{modelpkg}'")
        for model, md in pkginfo["models"].items():
            print(f"\t* '{model}': {md['description']}")
            print(f"\t\t - More info: {md['citation']}")

if __name__ == "__main__":
    print_models_info()