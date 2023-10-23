import enum
import json
import attrs
import cattrs
import tqdm.auto
import gtod.util


@attrs.frozen
class Slot:
    name: str
    description: str
    is_categorical: bool
    possible_values: list[str]


@attrs.frozen
class Intent:
    name: str
    description: str
    is_transactional: bool
    required_slots: list[str]
    optional_slots: list[str]
    result_slots: list[str] = attrs.field(factory=list)


@attrs.frozen
class Service:
    service_name: str
    description: str
    slots: list[Slot]
    intents: list[Intent]

    @property
    def name(self):
        return self.service_name


Schema = list[Service]


@enum.unique
class Speaker(gtod.util.AutoNameEnum):
    USER = enum.auto()
    SYSTEM = enum.auto()


@enum.unique
class UserAct(gtod.util.AutoNameEnum):
    INFORM_INTENT = enum.auto()
    NEGATE_INTENT = enum.auto()
    AFFIRM_INTENT = enum.auto()
    INFORM = enum.auto()
    REQUEST = enum.auto()
    AFFIRM = enum.auto()
    NEGATE = enum.auto()
    SELECT = enum.auto()
    REQUEST_ALTS = enum.auto()
    THANK_YOU = enum.auto()
    GOODBYE = enum.auto()


@enum.unique
class SystemAct(gtod.util.AutoNameEnum):
    INFORM = enum.auto()
    REQUEST = enum.auto()
    CONFIRM = enum.auto()
    OFFER = enum.auto()
    NOTIFY_SUCCESS = enum.auto()
    NOTIFY_FAILURE = enum.auto()
    INFORM_COUNT = enum.auto()
    OFFER_INTENT = enum.auto()
    REQ_MORE = enum.auto()
    GOODBYE = enum.auto()


@attrs.frozen
class SlotValue:
    slot: str
    start: int
    exclusive_end: int


@attrs.frozen(slots=False)
class DialogueAction:
    slot: str | None
    values: list[str] | None
    canonical_values: list[str] | None


@attrs.frozen(slots=False)
class UserAction(DialogueAction):
    act: UserAct


@attrs.frozen(slots=False)
class SystemAction(DialogueAction):
    act: SystemAct


@attrs.frozen(slots=False)
class DialogueFrame:
    service: str
    slots: list[SlotValue]


@attrs.frozen(slots=False)
class State:
    active_intent: str
    requested_slots: list[str]
    slot_values: dict[str, str]


@attrs.frozen(slots=False)
class UserFrame(DialogueFrame):
    actions: list[UserAction]
    state: State


@attrs.frozen
class ServiceCall:
    method: str
    parameters: dict[str, str]


@attrs.frozen(slots=False)
class SystemFrame(DialogueFrame):
    actions: list[SystemAction]


@attrs.frozen(slots=False)
class SystemFrameWithServiceCall(SystemFrame):
    service_call: ServiceCall | None
    service_results: list[dict[str, str]] | None


Frame = UserFrame | SystemFrame | SystemFrameWithServiceCall


@attrs.frozen
class Turn:
    speaker: Speaker
    utterance: str
    frames: list[Frame]


@attrs.frozen
class Dialogue:
    dialogue_id: str
    services: list[str]
    turns: list[Turn]


def load_dialogue_info(path):
    with path.joinpath("schema.json").open() as f:
        schema_json = json.load(f)
        schema = cattrs.structure(schema_json, Schema)

    registry = dict(zip([service.name for service in schema], schema))

    dialogues = []
    for dialogue_path in tqdm.auto.tqdm(path.glob("dialogues_*.json")):
        with dialogue_path.open() as f:
            dialogue_json = json.load(f)
            dialogues.extend(cattrs.structure(dialogue_json, list[Dialogue]))

    return schema, registry, dialogues


# sgd_path = pathlib.Path("../datasets/dstc8-schema-guided-dialogue").resolve()
# assert sgd_path.is_dir()

# split: pathlib.Path
# schema = dict()
# registry = dict()
# dialogues = dict()
# valid_splits = {"train", "test", "dev"}
# for split in sgd_path.iterdir():
#     if split.is_dir() and split.name in valid_splits:
#         schema[split.name], registry[split.name], dialogues[split.name] = load_dialogue_info(split)
