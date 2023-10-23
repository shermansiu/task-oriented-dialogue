# Copyright 2021 Google Research.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for processing multiwoz dialogue data.

# TODO: Add unit tests
"""

import collections
import collections.abc
import enum
import itertools
import json
import os
import pathlib
import attrs


# Use OrderedDict for JSON to preserve field order.
Json = collections.OrderedDict


class MultiwozVersion(str, enum.Enum):
    v21 = "2.1"
    v22 = "2.2"
    v23 = "2.3"
    v24 = "2.4"


@attrs.frozen
class MultiwozData:
    """Information from MultiWOZ dataset.

    Attributes:
        train_json: JSON for train dialogues.
        dev_json: JSON for dev dialogues.
        test_json: JSON for test dialogues.
        slot_descriptions: dict mapping slot name to list of slot descriptions.
    """

    train_json: Json
    dev_json: Json
    test_json: Json
    slot_descriptions: dict[str, list[str]]


@attrs.frozen
class SlotInfo:
    """Dataclass for information about a slot.

    Attributes:
        is_categorical: Whether this is a categorical or noncategorical slot.
        possible_values: A list of possible values. This is empty if this is a
            noncategorical slot.
    """

    is_categorical: bool
    possible_values: list[str]


@attrs.frozen
class SchemaInfo:
    """Dataclass for information from a schema.

    Attributes:
        slots_by_domain: slots_by_domain[domain][slot_name] has a SlotInfo dataclass
            for that particular domain and slot_name.
    """

    slots_by_domain: dict[str, dict[str, SlotInfo]]


def load_data(
    data_path: pathlib.Path, multiwoz_version: MultiwozVersion, is_trade: bool = False
) -> MultiwozData:
    """Loads MultiWOZ dataset.

    Args:
        data_path: Path to the multiwoz dataset.
        multiwoz_version: The version of the multiwoz dataset.
        is_trade: Whether the data is trade-preprocessed or not.

    Returns:
        A dataclass object storing the loaded dataset.
    """
    # Load dialogue data.
    if is_trade:
        with (data_path / "train_dials.json").open() as f:
            train_json = Json()
            for d in json.loads(f.read().lower(), object_pairs_hook=Json):
                train_json[d["dialogue_idx"]] = d

        with (data_path / "dev_dials.json").open() as f:
            dev_json = Json()
            for d in json.loads(f.read().lower(), object_pairs_hook=Json):
                dev_json[d["dialogue_idx"]] = d

        with (data_path / "test_dials.json").open() as f:
            test_json = Json()
            for d in json.loads(f.read().lower(), object_pairs_hook=Json):
                test_json[d["dialogue_idx"]] = d

    else:
        with (data_path / "data.json").open() as f:
            # Load using collections.OrderedDict to keep order the same as JSON.
            json_data = json.loads(f.read().lower(), object_pairs_hook=Json)

        # Different MultiWOZ versions have different (val|test)ListFile extensions
        # but both can be parsed as a text file containing a list of dialog ids.
        extension = "json" if multiwoz_version == "2.4" else "txt"
        with (data_path / f"valListFile.{extension}").open() as f:
            dev_ids = {line.lower().rstrip() for line in f}
        with (data_path / f"testListFile.{extension}").open() as f:
            test_ids = {line.lower().rstrip() for line in f}

        train_json, dev_json, test_json = Json(), Json(), Json()
        for dialog_idx, dialog_json in json_data.items():
            if dialog_idx in dev_ids:
                dev_json[dialog_idx] = dialog_json
            elif dialog_idx in test_ids:
                test_json[dialog_idx] = dialog_json
            else:
                train_json[dialog_idx] = dialog_json

    slot_descriptions = load_slot_descriptions(
        slot_descriptions_file_path=data_path / "slot_descriptions.json"
    )

    return MultiwozData(train_json, dev_json, test_json, slot_descriptions)


def load_slot_descriptions(
    slot_descriptions_file_path: pathlib.Path,
) -> dict[str, list[str]]:
    """Loads slot descriptions from Json file."""

    # Note that 2.4 doesn't come with a
    # slot_descriptions.json file. Copy the 2.1 file to avoid an error.
    with slot_descriptions_file_path.open() as f:
        slot_descriptions_raw = json.loads(f.read().lower(), object_pairs_hook=Json)
        slot_descriptions = {}
        for key, val in slot_descriptions_raw.items():
            # To be consistent with the keys from extract_belief_state(), rename
            # "book" slots. e.g. "hotel-book people" -> "hotel-people".
            key = key.replace("book ", "")
            # slot_descriptions.json has a "bus-arriveby" slot that doesn't actually
            # exist.
            if key in ("bus-arriveby", "bus-people"):
                continue

            slot_descriptions[key] = val
    return slot_descriptions


def load_schema(schema_path: pathlib.Path) -> SchemaInfo:
    """Load information from MultiWOZ schema file."""
    with schema_path.open() as f:
        schema_json = json.loads(f.read().lower(), object_pairs_hook=Json)

    slots_by_domain = {}
    for service in schema_json:
        domain = service["service_name"]
        slots_by_domain[domain] = {}
        for slot in service["slots"]:
            is_categorical = slot["is_categorical"]
            if is_categorical:
                possible_values = slot["possible_values"]
            else:
                possible_values = []

            # Don't consider numerical categorical slots as categorical.
            if is_categorical and all([_.isdigit() for _ in possible_values]):
                is_categorical = False
                possible_values = []

            # To be consistent with the slots in the dialogue, rename
            # "book" slots. e.g. "hotel-book people" -> "hotel-people".
            slot_name = slot["name"].replace("book", "")
            slots_by_domain[domain][slot_name] = SlotInfo(
                is_categorical=is_categorical, possible_values=possible_values
            )
    return SchemaInfo(slots_by_domain)


def get_domain(slot_name: str) -> str:
    """Extracts the domain from a Multiwoz slot name."""
    return slot_name.split("-")[0]


def extract_belief_state(metadata_json: Json, is_trade: bool) -> dict[str, str]:
    """Extracts belief states from data.

    Args:
        metadata_json: A json dict containing metadata about the dialogue.
        is_trade: Whether the data is trade-preprocessed or not.

    Returns:
        A mapping from slot name to value for the current dialogue.
    """
    state_dict = collections.OrderedDict()

    # Form belief state based on whether data is TRADE preprocessed or not
    if is_trade:
        for state in metadata_json:
            if len(state["slots"]) != 1:
                raise ValueError(
                    "Length of slots in state must be 1. Actual length: "
                    f"{len(state['slots'])}. "
                    f"state['slots']: {state['slots']}"
                )
            # To be consistent with the keys, rename
            # "book" slots. e.g. "hotel-book people" -> "hotel-people".
            slot_name = state["slots"][0][0].replace("book ", "")
            state_dict[slot_name] = state["slots"][0][1]
    else:
        for domain, state in metadata_json.items():
            # Two types of states: book and semi.
            domain_bs_book = state["book"]
            domain_bs_semi = state["semi"]
            # Note: "booked" is not really a state, just booking confirmation, and
            # val can be "dontcare".
            state_dict.update(
                (f"{domain}-{key}", val)
                for key, val in domain_bs_book.items()
                if val and val not in ("not mentioned", "none") and key != "booked"
            )
            state_dict.update(
                (f"{domain}-{key}", val)
                for key, val in domain_bs_semi.items()
                if val and val not in ("not mentioned", "none")
            )
    return state_dict


def extract_domains(belief_state: dict[str, str]) -> set[str]:
    """Extracts active domains in the dialogue state."""
    return set([get_domain(slot_name) for slot_name in belief_state.keys()])


# Dataclass representations of MultiWOZ dialogues.


@attrs.frozen
class MultiwozTurn:
    """A dataclass for one turn of a MultiWOZ dialogue.

    Attributes:
        utterance: The text utterance from a turn.
        belief_state: The slot-value pairs of the conversation.
    """

    utterance: str
    belief_state: dict[str, str]
    actions: dict[str, list[tuple[str, str]]]


@attrs.frozen
class MultiwozDialog:
    """A dataclass for a MultiWOZ dialogue.

    Attributes:
        dialog_id: The ID of the dialogue.
        turns: A list of MultiwozTurn's.
    """

    dialog_id: str
    turns: list[MultiwozTurn]


@attrs.frozen
class MultiwozDataclassData:
    train_dialogs: dict[str, MultiwozDialog]
    dev_dialogs: dict[str, MultiwozDialog]
    test_dialogs: dict[str, MultiwozDialog]
    slot_descriptions: dict[str, list[str]]

    def all_dialogs(self) -> collections.abc.Iterator[tuple[str, MultiwozDialog]]:
        return itertools.chain(
            self.train_dialogs.items(),
            self.dev_dialogs.items(),
            self.test_dialogs.items(),
        )

    def dialogs_by_split(
        self,
    ) -> collections.abc.Iterator[tuple[str, dict[str, MultiwozDialog]]]:
        yield from (
            ("train", self.train_dialogs),
            ("dev", self.dev_dialogs),
            ("test", self.test_dialogs),
        )


def load_data_as_dataclasses(
    data_path: pathlib.Path, multiwoz_version: MultiwozVersion, is_trade: bool = False
) -> MultiwozDataclassData:
    """Loads MultiWOZ dataset.

    Args:
        data_path: Path to the multiwoz dataset.
        multiwoz_version: The version of the multiwoz dataset.
        is_trade: Whether the data is trade-preprocessed or not.

    Returns:
        A dataclass object storing the loaded dataset.
    """
    multiwoz_data = load_data(data_path, multiwoz_version, is_trade)

    def _dataclass_from_json(json_data: Json) -> dict[str, MultiwozDialog]:
        dialogs = {}
        for dialog_id, dialog_json in json_data.items():
            turns = []
            for turn, utterance_json in enumerate(dialog_json["log"]):
                is_system = turn % 2 == 1
                speaker = "system" if is_system else "user"
                utterance = utterance_json["text"].strip().replace("\t", " ")
                belief_state = extract_belief_state(
                    metadata_json=utterance_json["metadata"], is_trade=False
                )
                actions = {}
                for act, slotvals in utterance_json["dialog_act"].items():
                    actions[act] = []
                    for slotval in slotvals:
                        assert len(slotval) == 2
                        actions[act].append(tuple(slotval))
                turns.append(MultiwozTurn(utterance, belief_state, actions))
            dialogs[dialog_id] = MultiwozDialog(dialog_id, turns)
        return dialogs

    train_dialogs = _dataclass_from_json(multiwoz_data.train_json)
    dev_dialogs = _dataclass_from_json(multiwoz_data.dev_json)
    test_dialogs = _dataclass_from_json(multiwoz_data.test_json)
    return MultiwozDataclassData(
        train_dialogs, dev_dialogs, test_dialogs, multiwoz_data.slot_descriptions
    )
