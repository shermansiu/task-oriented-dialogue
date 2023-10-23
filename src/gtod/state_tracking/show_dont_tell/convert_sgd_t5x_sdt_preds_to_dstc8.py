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

r"""Converts T5X predictions on SGD to DSTC8 official format for evaluation."""

import collections
import collections.abc
import json
import logging
import os
import pathlib
import re

import attrs
import tyro

from gtod.state_tracking.utils import sgd_utils
import gtod.util


@attrs.frozen
class ConvertSgdT5xSdtPredsToDstc8Config:
    """SDT predictions conversion CLI configuration.

    Attributes:
        t5x_predictions_jsonl: Input JSONL file with T5X model predictions.
        dstc8_data_dir: Directory for the downloaded DSTC8 data, which contains
            the dialogue files and schema files of all datasets (train, dev, test)
        output_dir: Output directory for JSON-format model predictions for official DSTC8
            evaluation.
        dataset_split: Dataset split for evaluation.
        delimiter: Delimiter to separate slot/intent IDs from their descriptions or values.
        evaluate_intent_acc: Whether to evaluate on active intent classification task.
    """

    t5x_predictions_jsonl: pathlib.Path
    dstc8_data_dir: pathlib.Path
    output_dir: pathlib.Path
    dataset_split: gtod.util.DatasetSplit = attrs.field(
        default=gtod.util.DatasetSplit.test
    )
    delimiter: str = attrs.field(default="=")
    evaluate_intent_acc: bool = attrs.field(default=False)


config = ConvertSgdT5xSdtPredsToDstc8Config(
    pathlib.Path("."), pathlib.Path("."), pathlib.Path(".")
)


_SDT_CAT_SLOT_IDENTIFIER = "of possible values"


def _create_categorical_slot_to_value_map(input_str: str) -> dict[str, dict[str, str]]:
    """Creates mappings from letters to values for categorical slots."""
    slot_values = (
        input_str.split("[slots]")[1].split("[context]")[0].split("[intent]")[0].strip()
    )
    slot_to_option_to_value = collections.defaultdict(dict)
    for slot, value in re.findall(
        rf"(\w+){config.delimiter}(.*?)(?=\w+{config.delimiter}|$)", slot_values
    ):
        if _SDT_CAT_SLOT_IDENTIFIER not in value:
            continue
        options_str = value.split(_SDT_CAT_SLOT_IDENTIFIER)[1].strip()
        for option, option_value in re.findall(
            r"([a-z])\) (.*?)(?=[a-z]\)|$)", options_str
        ):
            slot_to_option_to_value[slot][option] = option_value.strip()

    return slot_to_option_to_value


def _create_intent_map(input_str: str) -> dict[str, str]:
    """Creates mappings from letters to intent names."""
    intent_str = input_str.split("[intent]")[1].split("[context]")[0].strip()
    intent_option_to_value = {}
    if _SDT_CAT_SLOT_IDENTIFIER not in intent_str:
        raise ValueError("Improperly formatted intent prompt: %s" % intent_str)
    intent_str = intent_str.split(_SDT_CAT_SLOT_IDENTIFIER)[1].strip()
    for option, option_value in re.findall(r"([a-z])\) (.*?)(?=[a-z]\)|$)", intent_str):
        intent_option_to_value[option] = option_value.strip()

    return intent_option_to_value


def _normalize_value_prediction(
    slot_name: str, value: str, slot_to_option_to_value: dict[str, dict[str, str]]
) -> str | None:
    """Normalizes a predicted value and maps a categorical option to value."""
    value = value.strip()
    if value == "none":
        return None

    # Map decoded multiple choice letters back to actual value for cat slots.
    elif slot_name in slot_to_option_to_value:
        if value in slot_to_option_to_value[slot_name]:
            value = slot_to_option_to_value[slot_name][value]
        # Print cases where model didn't decode a valid multiple choice letter.
        elif value != "dontcare":
            logging.info(
                "Unexpected slot scenario. slot_name %s. value %s. "
                "slot_to_option_to_value %s",
                slot_name,
                value,
                slot_to_option_to_value,
            )

    return value


def populate_json_predictions(
    dialog_id_to_dialogue: dict[str, sgd_utils.DialoguesDict],
    frame_predictions: dict[str, str | dict[str, str]],
) -> None:
    """Populates a dialogue JSON dictionary with frame-level T5X model outputs.

    Given a single prediction from frame_predictions, this looks up the
    corresponding frame from dialog_id_to_dialogue and modifies it in-place by
    inserting the predictions into the dialogue state field.

    Args:
        dialog_id_to_dialogue: A mapping from dialog id to the dialogue json object
        frame_predictions: A dict containing T5X predictions and example metadata
    """
    preds = frame_predictions["prediction"]
    if not isinstance(preds, str):
        raise ValueError(
            f"'preds' must be string type, " f"not {type(preds)}. preds: {preds}"
        )
    dialog_id = frame_predictions["input"]["dialogue_id"]
    turn_id = int(frame_predictions["input"]["turn_id"])
    frame_id = int(frame_predictions["input"]["frame_id"])

    if dialog_id not in dialog_id_to_dialogue:
        raise ValueError(f"Dialogue ID {dialog_id} not found.")

    frame = dialog_id_to_dialogue[dialog_id]["turns"][turn_id]["frames"][frame_id]

    input_str = frame_predictions["input"]["inputs_pretokenized"]

    # Create a dict(slot -> dict(multiple-choice letter -> value)) for cat slots.
    slot_to_option_to_value = _create_categorical_slot_to_value_map(input_str)

    if config.evaluate_intent_acc:
        # Create a dict(multiple-choice letter -> intent) for intents.
        option_to_intent = _create_intent_map(input_str)

    # Read and populate all slot value predictions.
    # TODO(harrisonlee): Support requested slots.
    slot_preds = preds.split("[state]")[1].split("[intent]")[0].strip()
    for slot_name, value in re.findall(
        rf"(\w+){config.delimiter}(.*?)(?=\w+{config.delimiter}|$)", slot_preds
    ):
        value = _normalize_value_prediction(slot_name, value, slot_to_option_to_value)

        if value:
            frame["state"]["slot_values"][slot_name] = [value]

    # Populate intent prediction.
    if config.evaluate_intent_acc and "[intent]" in preds:
        # Read and populate intent prediction.
        intent_pred = preds.split("[intent]")[1].strip()
        frame["state"]["active_intent"] = option_to_intent.get(intent_pred, "NONE")


def main() -> None:
    assert config is not None

    # Load dialogues and flatten into dict(dialogue_id->dialogue).
    subdir_to_dialogues = {}
    sgd_utils.load_dialogues_to_dict(
        config.dstc8_data_dir, config.dataset_split, subdir_to_dialogues
    )
    dialog_id_to_dialogue = {}
    for dialogues in subdir_to_dialogues[config.dataset_split].values():
        for dialog in dialogues:
            dialog_id_to_dialogue[dialog["dialogue_id"]] = dialog

    # Erase ground truth state values.
    for dial in dialog_id_to_dialogue.values():
        for turn in dial["turns"]:
            for frame in turn["frames"]:
                if "state" in frame:
                    frame["state"]["slot_values"] = {}
                    frame["state"]["requested_slots"] = []
                    frame["state"]["active_intent"] = "NONE"

    # Read JSONL predictions.
    with config.t5x_predictions_jsonl.open("r") as predictions_file:
        for line in predictions_file:
            frame_predictions = json.loads(line)
            populate_json_predictions(dialog_id_to_dialogue, frame_predictions)

    # Write JSON predictions.
    output_dir = config.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    with (output_dir / "dialogues_all.json").open("w") as output_file:
        json.dump(
            list(dialog_id_to_dialogue.values()),
            output_file,
            indent=2,
            separators=(",", ": "),
        )


if __name__ == "__main__":
    config = tyro.cli(ConvertSgdT5xSdtPredsToDstc8Config)
    main()
