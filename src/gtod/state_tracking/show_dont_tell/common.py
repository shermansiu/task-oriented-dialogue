import enum

import gtod.util


class PromptFormat(gtod.util.AutoNameEnum):
    """Format of the prompt for priming.

    "separated" means a dialogue followed by a separate string of slots.
    """

    separated = enum.auto()


class ContextFormat(gtod.util.AutoNameEnum):
    """Format of the dialogue context."""

    dialogue = enum.auto()


class TargetFormat(gtod.util.AutoNameEnum):
    """Format of the target.

    "all" and "active" respectively refer to all and only active slots being present in the target.
    """

    all = enum.auto()
    active = enum.auto()
