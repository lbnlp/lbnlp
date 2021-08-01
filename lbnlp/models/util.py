import pip
import warnings
import pickle


class ModelRequirementError(BaseException):
    pass


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def check_versions(reqs):
    suffix = f"Full requirements are: {reqs}"
    for req in reqs:
        if "==" in req:
            req_split = req.split("==")
            if len(req_split) != 2:
                raise ModelRequirementError(
                    f"Version of '{req}' is misformatted, version must be specified. {suffix}")
            else:
                req_name = req_split[0]
                req_version = req_split[1]

                try:
                    mod = __import__(req_name)
                except ImportError:
                    raise ModelRequirementError(
                        f"Requirement {req_name} with required version {req_version} is not installed. {suffix}")

                try:
                    installed_version = mod.__version__

                    if installed_version != req_version:
                        raise ModelRequirementError(
                            f"Requirement {req_name} has required version {req_version}, you have {installed_version}. {suffix}")
                except AttributeError:
                    warnings.warn(
                        f"Requirement {req_name} has no version tag! There is no way to ensure the correct version is installed for this model! {suffix}")

        elif "git" in req:
            warnings.warn(f"Requirement {req} must be installed for this model package to function. Please ensure this package from git is installed from source. {suffix}")
        else:
            raise ModelRequirementError(
                f"Exact version of model requirement '{req}' not specified with '==', e.g., 'sklearn==0.19.0' or with git. {suffix}")


def model_loader_setup(pkg):
    def decorator(loader_func):
        def wrapper(model_name, ignore_requirements=False):
            pkg.load()
            if model_name not in pkg.model_names:
                raise ValueError(f"Model {model_name} in {pkg.modelpkg_name} not found. Choose from {pkg.model_names}")
            if not ignore_requirements:
                check_versions(pkg.models_info[model_name]["requirements"])

            return loader_func(model_name, ignore_requirements)
        return wrapper
    return decorator