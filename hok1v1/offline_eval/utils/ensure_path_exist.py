import os
import os.path as osp


def ensure_path_exist(path: str) -> None:
    """
    Ensure the given path exist otherwise create the path.
    Args:
        path (str):
            The path to be checked.
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

    assert osp.exists(path), '{} does not exist'.format(path)
