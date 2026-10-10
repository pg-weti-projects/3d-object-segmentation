from pathlib import Path
import tomllib


def load_settings() -> dict:
    """Load application settings from the local TOML file."""
    config_path = Path(__file__).with_name("config.toml")
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file).get("app", {})
