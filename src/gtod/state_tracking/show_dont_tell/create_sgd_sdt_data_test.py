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

"""Tests for create_sgd_sdt_data."""

import contextlib
import sys
import pytest
import gtod.util
from gtod.state_tracking.show_dont_tell import create_sgd_sdt_data


config_saver = gtod.util.config_saver_factory(create_sgd_sdt_data)


@pytest.mark.parametrize(
    "target_format, use_slot_ids, mcq_cat_vals, use_intent_slot_descs, ref_output_filename",
    [
        (
            "all_slots_slot_names",
            "all",
            False,
            False,
            False,
        ),
        (
            "active_slots_slot_names",
            "active",
            False,
            False,
            False,
        ),
        (
            "all_slots_slot_ids",
            "all",
            True,
            False,
            False,
        ),
        (
            "all_slots_cat_val_mcq",
            "all",
            False,
            True,
            False,
        ),
        (
            "all_slots_d3st_mcq",
            "all",
            False,
            True,
            True,
        ),
        (
            "all_slots_slot_ids_d3st_mcq",
            "all",
            True,
            True,
            True,
        ),
    ],
    ids=[
        "sgd_text_sdt_separated_dialogue_all",
        "sgd_text_sdt_separated_dialogue_active",
        "sgd_text_sdt_separated_dialogue_all_slot_ids",
        "sgd_text_sdt_separated_dialogue_all_cat_val_mcq",
        "sgd_text_sdt_separated_dialogue_all_mcq_d3st",
        "sgd_text_sdt_separated_dialogue_all_slot_ids_mcq_d3st",
    ],
)
def test_generate_data(
    target_format,
    use_slot_ids,
    mcq_cat_vals,
    use_intent_slot_descs,
    ref_output_filename,
    tmp_path,
    testdata_dir,
):
    temp_output = tmp_path / "output"
    ref_output = testdata_dir / "show_dont_tell" / ref_output_filename

    with config_saver(
        input_dir=testdata_dir / "sgd_data",
        output_path=temp_output,
        subdirs=["train"],
        prompt_format="separated",
        prompt_indices=[0],
        context_format="dialogue",
        target_format=target_format,
        add_intents=False,
        use_slot_ids=use_slot_ids,
        randomize_slots=False,
        randomize_intents=False,
        mcq_cat_vals=mcq_cat_vals,
        mcq_intents=False,
        randomize_cat_vals=False,
        use_intent_slot_descs=use_intent_slot_descs,
    ):
        create_sgd_sdt_data.main()

    with temp_output.open() as temp_f, ref_output.open() as ref_f:
        assert temp_f.readlines() == ref_f.readlines()


def test_generate_sgdx_data(tmp_path, testdata_dir):
    temp_output = tmp_path / "output"
    ref_output = (
        testdata_dir / "show_dont_tell" / "sgd_text_sdt_separated_dialogue_all_sgdx"
    )

    with config_saver(
        input_dir=testdata_dir / "sgd_data",
        output_path=temp_output,
        sgdx_dir=testdata_dir / "sgdx_data",
        subdirs=["train"],
        prompt_format="separated",
        prompt_indices=[0],
        context_format="dialogue",
        target_format="all",
        add_intents=False,
        use_slot_ids=False,
        randomize_slots=False,
        randomize_intents=False,
        mcq_cat_vals=False,
        mcq_intents=False,
        randomize_cat_vals=False,
    ):
        create_sgd_sdt_data.main()

    with temp_output.open() as temp_f, ref_output.open() as ref_f:
        assert temp_f.readlines() == ref_f.readlines()


@pytest.mark.parametrize(
    "mcq_intents, ref_output_filename",
    [
        (
            "all_slots_slot_names_intent",
            False,
        ),
        (
            "all_slots_slot_names_intent_mcq",
            True,
        ),
    ],
    ids=[
        "sgd_text_sdt_separated_dialogue_all_intent",
        "sgd_text_sdt_separated_dialogue_all_intent_mcq",
    ],
)
def test_generate_intent(
    self, mcq_intents, ref_output_filename, tmp_path, testdata_dir
):
    temp_output = tmp_path / "output"
    ref_output = testdata_dir / "show_dont_tell" / ref_output_filename

    with config_saver(
        input_dir=testdata_dir / "sgd_data",
        output_path=temp_output,
        subdirs=["train"],
        prompt_format="separated",
        prompt_indices=[0],
        context_format="dialogue",
        target_format="all",
        add_intents=True,
        use_slot_ids=False,
        randomize_slots=False,
        randomize_intents=False,
        mcq_cat_vals=False,
        mcq_intents=mcq_intents,
        randomize_cat_vals=False,
    ):
        create_sgd_sdt_data.main()

    with temp_output.open() as temp_f, ref_output.open() as ref_f:
        self.assertEqual(temp_f.readlines(), ref_f.readlines())


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"]))
