import os
import json
import hashlib
import requests


data_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(data_dir, "pkg")
models_dir = os.path.join(data_dir, "models")

with open("metadata.json", "r") as f:
    metadata = json.load(f)

metadata_modelpkgs = metadata["modelpkgs"]
metadata_models = metadata["models"]


def get_available_models():
    return list(metadata_models.keys())


def get_available_modelpkgs():
    return list(metadata_modelpkgs.keys())


def _download_modelpkg(modelpkg_name):

    if not os.path.exists(pkg_dir):
        os.makedirs(pkg_dir)

    if modelpkg_name not in metadata_modelpkgs:
        raise KeyError(f"Model package name {modelpkg_name} not in metadata")

    md = metadata_modelpkgs[modelpkg_name]
    file_path = os.path.join(pkg_dir, modelpkg_name)
    url = md["url"]

    # Download modelpkg only if not already downloaded.
    if os.path.exists(file_path):
        was_downloaded = False
    else:
        print(f"Fetching {os.path.basename(file_path)} model package from {url} to {file_path}", flush=True)
        r = requests.get(url, stream=True)
        with open(file_path, "wb") as file_out:
            for chunk in r.iter_content(chunk_size=2048):
                file_out.write(chunk)
        r.close()
        was_downloaded = True

    sha256_test = _get_file_sha256_hash(file_path)
    sha256_truth = md["sha256"]
    if sha256_test != sha256_truth:
        raise ValueError(f"Hash of modelpkg file {os.path.basename(file_path)} ({sha256_test}) does not match truth hash ({sha256_truth}).")
    return was_downloaded

def _get_file_sha256_hash(file_path):
    """
    Takes a file and returns the SHA 256 hash of its data

    Args:
        file_path (str): path of file to hash

    Returns: (str)

    """
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(file_path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()



if __name__ == "__main__":
    _download_modelpkg("matscholar_2020v1")