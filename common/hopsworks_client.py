import hopsworks

from common.config import require_env

_project = None


def get_project():
    global _project
    if _project is None:
        _project = hopsworks.login(api_key_value=require_env("HOPSWORKS_API_KEY"))
    return _project


def get_feature_store():
    return get_project().get_feature_store()


def get_model_registry():
    return get_project().get_model_registry()
