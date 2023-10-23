import enum

import gtod.util


class DescriptionType(gtod.util.AutoNameEnum):
    """What to use for the slot descriptions.

    full_desc: A natural language description.
    full_desc_with_domain: Domain, followed by natural language description.
    item_name: The name of the slot.
    shuffled_item_name: Random permutation of the slot name.
    """

    full_desc = enum.auto()
    full_desc_with_domain = enum.auto()
    item_name = enum.auto()
    shuffled_item_name = enum.auto()


class MultipleChoiceFormat(gtod.util.AutoNameEnum):
    none = enum.auto()
    a = enum.auto()
    one_a = "1a"
