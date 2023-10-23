# Copyright 2021 Google Research.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for convert_sgd_t5x_sdt_preds_to_dstc8."""

import copy
import json
import pathlib
import sys

import pytest

from gtod.state_tracking.show_dont_tell import common
from gtod.state_tracking.show_dont_tell import convert_sgd_t5x_sdt_preds_to_dstc8


DIALOG_ID_TO_DIALOGUE = {
    "1_00000": {
        "dialogue_id": "1_00000",
        "services": ["Restaurants_2"],
        "turns": [
            {
                "frames": [
                    {
                        "service": "Restaurants_2",
                        "slots": [{"exclusive_end": 52, "slot": "date", "start": 45}],
                        "state": {
                            "active_intent": "",
                            "requested_slots": [],
                            "slot_values": {},
                        },
                    }
                ],
                "speaker": "USER",
                "utterance": "Hi, could you get me a vegetarian restaurant booking on "
                "the 8th please?",
            }
        ],
    }
}


@pytest.fixture
def dummy_config():
    return convert_sgd_t5x_sdt_preds_to_dstc8.CliConfig(
        pathlib.Path("."),
        pathlib.Path("."),
        pathlib.Path("."),
        evaluate_intent_acc=True,
    )


def test_convert_data(tmp_path, testdata_dir):
    config = convert_sgd_t5x_sdt_preds_to_dstc8.CliConfig(
        t5x_predictions_jsonl=testdata_dir
        / "show_dont_tell"
        / "sgd_t5x_prediction.jsonl",
        dstc8_data_dir=testdata_dir / "sgd_data",
        output_dir=tmp_path,
        dataset_split=common.DatasetSplit.dev,
    )
    convert_sgd_t5x_sdt_preds_to_dstc8.main(config)

    with (tmp_path / "dialogues_all.json").open() as f:
        dialogues = json.load(f)
    actual_dialogue_slots = dialogues[0]["turns"][0]["frames"][0]["state"][
        "slot_values"
    ]
    expected_dialogue_slots = {
        "number_of_seats": ["2"],
        "time": ["half past 11 in the morning"],
    }

    assert actual_dialogue_slots == expected_dialogue_slots


def test_populate_json_slot_predictions(dummy_config):
    frame_predictions = {
        "input": {
            "inputs_pretokenized": "[example] [user] how about finding a place march 3rd? "
            "somewhere moderate in cost that has vegetarian menu items. "
            "[system] i assume in novato? [user] novato is correct. "
            "something serving latin american cuisine and if possible with"
            " outdoor seating. [system] i found 1 called maya palenque "
            "restaurant. [user] i bet they have good food. [system] should"
            " i book you a table? [user] yes for two please. [system] what"
            " time would you like it for? [user] in the morning 11:15 "
            "please. [slots] number_of_seats=c of possible values a) 4 b) "
            "1 c) 2 d) 3 e) 6 f) 5 has_vegetarian_options=b of possible "
            "values a) false b) true restaurant_name=maya palenque "
            "restaurant date=march 3rd location=novato price_range=d of "
            "possible values a) cheap b) pricey c) ultra high-end d) "
            "moderate time=morning 11:15 has_seating_outdoors=b of "
            "possible values a) false b) true category=latin american "
            "[context] [user] hi, could you get me a "
            "vegarian restaurant booking on the 8th please?",
            "dialogue_id": "1_00000",
            "turn_id": "0",
            "frame_id": "0",
        },
        "prediction": "[state] number_of_seats=none has_vegetarian_options=b "
        "restaurant_name=none date=the 8th location=none price_range=none "
        "time=none has_seating_outdoors=none category=none",
    }
    dialog_id_to_dialogue = copy.deepcopy(DIALOG_ID_TO_DIALOGUE)
    convert_sgd_t5x_sdt_preds_to_dstc8.populate_json_predictions(
        dummy_config, dialog_id_to_dialogue, frame_predictions
    )
    actual_dialogue_slots = dialog_id_to_dialogue["1_00000"]["turns"][0]["frames"][0][
        "state"
    ]["slot_values"]
    expected_dialogue_slots = {
        "has_vegetarian_options": ["true"],
        "date": ["the 8th"],
    }
    assert actual_dialogue_slots == expected_dialogue_slots


def test_populate_json_intent_predictions(dummy_config):
    frame_predictions = {
        "input": {
            "inputs_pretokenized": "[example] [user] how about finding a place march 3rd? "
            "somewhere moderate in cost that has vegetarian menu items. "
            "[system] i assume in novato? [user] novato is correct. "
            "something serving latin american cuisine and if possible with"
            " outdoor seating. [system] i found 1 called maya palenque "
            "restaurant. [user] i bet they have good food. [system] should"
            " i book you a table? [user] yes for two please. [system] what"
            " time would you like it for? [user] in the morning 11:15 "
            "please. [slots] number_of_seats=c of possible values a) 4 b) "
            "1 c) 2 d) 3 e) 6 f) 5 [intent] a of possible options a) "
            "reserverestaurant b) findrestaurants [context] [user] hi, "
            "could you get me a vegarian restaurant booking on the 8th "
            "please?",
            "dialogue_id": "1_00000",
            "turn_id": "0",
            "frame_id": "0",
        },
        "prediction": "[state] number_of_seats=none [intent] a",
    }
    dialog_id_to_dialogue = copy.deepcopy(DIALOG_ID_TO_DIALOGUE)
    convert_sgd_t5x_sdt_preds_to_dstc8.populate_json_predictions(
        dummy_config, dialog_id_to_dialogue, frame_predictions
    )

    actual_intent = dialog_id_to_dialogue["1_00000"]["turns"][0]["frames"][0]["state"][
        "active_intent"
    ]
    expected_intent = "ReserveRestaurant"
    assert actual_intent.lower() == expected_intent.lower()


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"]))
