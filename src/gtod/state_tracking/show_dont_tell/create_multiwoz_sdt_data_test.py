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

"""Tests for create_multiwoz_sdt_data."""

import json
import shutil
import sys

import pytest

from gtod.state_tracking.show_dont_tell import create_multiwoz_sdt_data
from gtod.state_tracking.show_dont_tell import common
from gtod.state_tracking.utils import multiwoz_utils


@pytest.fixture(autouse=True)
def setup(tmp_path, testdata_dir):
    schema_file = tmp_path / "schema.json"

    # Setup data - general
    shutil.copy(
        testdata_dir / "multiwoz_slot_descriptions.json",
        tmp_path / "slot_descriptions.json",
    )
    shutil.copy(
        testdata_dir / "multiwoz_schema_schemaless.json",
        schema_file,
    )

    # Setup data - for 2.1 TRADE preprocessed data
    for target_file in ["train_dials.json", "dev_dials.json", "test_dials.json"]:
        shutil.copy(
            testdata_dir / "multiwoz_data_trade.json",
            tmp_path / target_file,
        )

    # Setup data - for non-TRADE 2.2-2.4 data
    shutil.copy(
        testdata_dir / "multiwoz_data.json",
        tmp_path / "data.json",
    )

    # Touch empty files for (val|test)ListFile.json
    (tmp_path / "valListFile.json").touch()
    (tmp_path / "testListFile.json").touch()


@pytest.mark.parametrize(
    "use_active_domains_only, blocked_domains, mcq_cat_vals, multiwoz_version, ref_output_filename, expected_len",
    [
        (
            False,
            set(),
            False,
            "2.4",
            "mw24_sdt_all_domains.json",
            7,
        ),
        (
            True,
            set(),
            False,
            "2.4",
            "mw24_sdt_active_domains.json",
            7,
        ),
        (
            False,
            set(),
            True,
            "2.4",
            "mw24_sdt_all_domains_cat_val_mcq.json",
            7,
        ),
        (
            True,
            {"hotel"},
            False,
            "2.4",
            "mw24_sdt_active_domains_block_hotel.json",
            4,
        ),
        (
            False,
            set(),
            False,
            "2.1",
            "mw21_trade_sdt_all_domains.json",
            7,
        ),
    ],
    ids=[
        "all_domains",
        "active_domains",
        "all_domains_cat_val_mcq",
        "active_domains_block_restaurant",
        "all_domains_21",
    ],
)
def test_generate_data(
    use_active_domains_only,
    blocked_domains,
    mcq_cat_vals,
    multiwoz_version,
    ref_output_filename,
    expected_len,
    tmp_path,
    testdata_dir,
):
    ref_output = testdata_dir / "show_dont_tell" / ref_output_filename

    is_trade = True if multiwoz_version == "2.1" else False
    multiwoz_data = multiwoz_utils.load_data(tmp_path, multiwoz_version, is_trade)
    examples = create_multiwoz_sdt_data.create_sdt_examples(
        multiwoz_data.train_json,
        create_multiwoz_sdt_data.Options(
            multiwoz_version=multiwoz_utils.MultiwozVersion(multiwoz_version),
            is_trade=is_trade,
            prompt_format=common.PromptFormat.separated,
            prompt_indices=[0],
            context_format=common.ContextFormat.dialogue,
            target_format=common.TargetFormat.all,
            randomize_slots=False,
            use_active_domains_only=use_active_domains_only,
            blocked_domains=blocked_domains,
            mcq_cat_vals=mcq_cat_vals,
            randomize_cat_vals=False,
            lowercase=True,
        ),
    )

    assert len(examples) == expected_len

    # Reference json contains 2nd and last examples we expect to generate
    with ref_output.open() as ref_f:
        ref_json = json.load(ref_f)

    # Compare the 2nd example generated
    assert examples[1].src == ref_json[0]["src"]
    assert examples[1].tgt == ref_json[0]["tgt"]
    assert examples[1].dialog_id == ref_json[0]["dialog_id"]
    assert examples[1].turn == ref_json[0]["turn"]

    # Compare the last example generated
    assert examples[-1].src == ref_json[1]["src"]
    assert examples[-1].tgt == ref_json[1]["tgt"]
    assert examples[-1].dialog_id == ref_json[1]["dialog_id"]
    assert examples[-1].turn == ref_json[1]["turn"]


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"]))
