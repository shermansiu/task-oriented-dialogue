import enum


class AutoNameEnum(str, enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self):
        return self.value
