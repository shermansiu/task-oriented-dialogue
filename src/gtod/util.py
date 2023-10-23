import enum


class AutoNameEnum(str, enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class DatasetSplit(AutoNameEnum):
    train = enum.auto()
    dev = enum.auto()
    test = enum.auto()
