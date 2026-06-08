import yaml


def load_config(base_path: str, override_path: str | None = None) -> dict:
    with open(base_path) as f:
        cfgs = yaml.safe_load(f)

    if override_path:
        with open(override_path) as o:
            overrides = yaml.safe_load(o)

        cfgs.update(overrides)

    return cfgs


if __name__ == "__main__":
    # x = load_config("/home/sk/Desktop/gacf/configs/base.yaml")
    x = load_config("./configs/base.yaml")
    print("values", x)
