from pathlib import Path


class Config(object):
    ROOT_DIR = Path(__file__).parent.absolute()
    DATA_DIR = ROOT_DIR / 'data'
