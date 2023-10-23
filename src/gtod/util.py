import contextlib
import enum
import types


class AutoNameEnum(str, enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class DatasetSplit(AutoNameEnum):
    train = enum.auto()
    dev = enum.auto()
    test = enum.auto()


def config_saver_factory(module: types.ModuleType):
    Config = type(module.config)

    @contextlib.contextmanager
    def config_saver(*args, **kwargs):
        old_config = module.config
        try:
            setattr(module, "config", Config(*args, **kwargs))
            yield
        finally:
            setattr(module, "config", old_config)

    return config_saver
