"""Language descriptions for offline GRAIL episodes."""

import re


DEFAULT_TASK_TEMPLATE = (
    "Pick up the {object} from the table{turn} and release it above the red bin."
)


def describe_task(template, object_key, turn_command):
    asset_name = re.fullmatch(r".+?__(.+)_\d+__\d+", object_key)
    object_name = (asset_name.group(1) if asset_name else object_key).replace("_", " ")
    if turn_command > 0:
        turn = ", turn around to the left,"
    elif turn_command < 0:
        turn = ", turn around to the right,"
    else:
        turn = ""
    return template.replace("{object}", object_name).replace("{turn}", turn)
