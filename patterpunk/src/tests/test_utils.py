import os

DIRECTORY = os.path.dirname(__file__)
RESOURCES = f'{DIRECTORY}/resources'


def get_resource(resource: str) -> str:
    return f'{RESOURCES}/{resource}'
