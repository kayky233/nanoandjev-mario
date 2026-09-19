"""Create a Mario environment across the 7.x Gym and 9.x Gymnasium stacks."""

from importlib.metadata import PackageNotFoundError, version


def make_mario_env(gym_super_mario_bros, level: str):
    """Return an env with the modern reset/step API expected by the harnesses."""
    try:
        major = int(version("gym-super-mario-bros").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        major = 9
    kwargs = {"apply_api_compatibility": True} if major < 9 else {}
    return gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0", **kwargs)
