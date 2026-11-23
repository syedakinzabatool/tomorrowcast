"""Run once to confirm the Hopsworks connection works: python scripts/smoke_test_hopsworks.py"""
from common.hopsworks_client import get_feature_store


def main() -> None:
    fs = get_feature_store()
    print(f"Connected OK. Feature store: {fs.name} (project: {fs.project_name})")


if __name__ == "__main__":
    main()
