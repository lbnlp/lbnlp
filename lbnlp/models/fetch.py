import os
import json
import hashlib
import zipfile
import requests


class ModelPkgLoader:
    def __init__(self, modelpkg_name):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.pkg_dir = os.path.join(self.data_dir, "pkg")
        self.models_dir = os.path.join(self.data_dir, "models")
        self.metadata_path = os.path.join(self.data_dir, "modelpkg_metadata.json")

        with open(self.metadata_path, "r") as f:
            self.metadata_modelpkgs = json.load(f)

        if modelpkg_name not in self.metadata_modelpkgs:
            raise KeyError(
                f"Model package name {modelpkg_name} not in metadata")

        self.metadata_pkg = self.metadata_modelpkgs[modelpkg_name]
        self.model_requirements = self.metadata_pkg["requirements"]
        self.model_names = list(self.model_requirements.keys())
        self.file_path = os.path.join(self.pkg_dir, modelpkg_name)
        self.structured_path = os.path.join(self.models_dir, modelpkg_name)

        self.modelpkg_name = modelpkg_name

        self.is_downloaded = None

    def download(self):
        """
        Fetch the raw model package file from an online repo.

        Returns:

        """
        if not os.path.exists(self.pkg_dir):
            os.makedirs(self.pkg_dir)

        url = self.metadata_pkg["url"]

        # Download modelpkg only if not already downloaded.
        if os.path.exists(self.file_path):
            self.is_downloaded = True
        else:
            print(f"Fetching {os.path.basename(self.file_path)} model package from {url} to {self.file_path}", flush=True)
            r = requests.get(url, stream=True)
            with open(self.file_path, "wb") as file_out:
                for chunk in r.iter_content(chunk_size=2048):
                    file_out.write(chunk)
            r.close()
            self.is_downloaded = True

    def validate(self):
        """
        Ensure the raw file hash matches the canonical version.
        Returns:

        """
        print("Validating ")
        sha256_test = _get_file_sha256_hash(self.file_path)
        sha256_truth = self.metadata_pkg["hash"]
        if sha256_test != sha256_truth:
            raise ValueError(
                f"Hash of modelpkg file {os.path.basename(self.file_path)} ({sha256_test}) does not match truth hash ({sha256_truth}).")

    def structure(self):
        """
        Move the models into the models dir in a logical format.

        Returns:

        """
        if self.modelpkg_name in ["matscholar_2020v1"]:
            with zipfile.ZipFile(self.file_path, "r") as zipped:
                zipped.extractall(self.structured_path)
        else:
            raise NotImplementedError(
                f"Model package {self.modelpkg_name} has no structuring/unzipping protocol")

    def load(self):
        if not os.path.exists(self.structured_path):
            self.download()
            self.validate()
            self.structure()



def _get_file_sha256_hash(file_path):
    """
    Takes a file and returns the SHA256 hash of its data

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

