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


def get_or_create_feature_group(name: str, version: int, primary_key: list[str], event_time: str, description: str):
    return get_feature_store().get_or_create_feature_group(
        name=name,
        version=version,
        primary_key=primary_key,
        event_time=event_time,
        description=description,
        online_enabled=False,
        time_travel_format="HUDI",
    )
