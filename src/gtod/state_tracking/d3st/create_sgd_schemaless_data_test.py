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

"""Tests for SGD text data generation."""

import contextlib
import filecmp
import pathlib
import sys

import pytest

from gtod.state_tracking.d3st import create_sgd_schemaless_data
from gtod.state_tracking.d3st import common


@pytest.mark.parametrize(
    "level, data_format",
    [
        ("dst", "full_desc"),
        ("dst_intent", "full_desc"),
    ],
    ids=["create_sgd_schemaless_dst", "create_sgd_schemaless_dst_intent"],
)
def test_generate_data_full_desc(
    level, data_format, tmp_path: pathlib.Path, testdata_dir: pathlib.Path
):
    temp_output = tmp_path / "output"
    ref_output = testdata_dir / f"sgd_text_v2_full_desc_{level}"

    config = create_sgd_schemaless_data.CliConfig(
        level=common.GenerationLevel(level),
        data_format=common.DataFormat(data_format),
        delimiter="=",
        sgd_file=testdata_dir / "sgd_train.json",
        schema_file=testdata_dir / "sgd_train_schema.json",
        output_file=temp_output,
        randomize_items=False,
    )
    slots, item_desc = create_sgd_schemaless_data.load_schema(config)
    create_sgd_schemaless_data.generate_data(config, slots, item_desc)
    assert filecmp.cmp(temp_output, ref_output)


@pytest.mark.parametrize(
    "level, data_format",
    [
        ("dst", "item_name"),
        ("dst_intent", "item_name"),
    ],
    ids=["create_sgd_schemaless_dst", "create_sgd_schemaless_dst_intent"],
)
def test_generate_data_item_name(level, data_format, tmp_path, testdata_dir):
    temp_output = tmp_path / "output"
    ref_output = testdata_dir / f"sgd_text_v2_item_name_{level}"

    config = create_sgd_schemaless_data.CliConfig(
        level=common.GenerationLevel(level),
        data_format=common.DataFormat(data_format),
        delimiter="=",
        sgd_file=testdata_dir / "sgd_train.json",
        schema_file=testdata_dir / "sgd_train_schema.json",
        output_file=temp_output,
        randomize_items=False,
    )
    slots, item_desc = create_sgd_schemaless_data.load_schema(config)
    create_sgd_schemaless_data.generate_data(config, slots, item_desc)
    assert filecmp.cmp(temp_output, ref_output)


@pytest.mark.parametrize(
    "level, data_format",
    [
        ("dst", "full_desc"),
        ("dst_intent", "full_desc"),
    ],
    ids=["create_sgd_schemaless_dst", "create_sgd_schemaless_dst_intent"],
)
def test_multiple_choice(level, data_format, tmp_path, testdata_dir):
    temp_output = tmp_path / "output"
    ref_output = testdata_dir / f"sgd_text_v2_multiple_choice_{level}"

    config = create_sgd_schemaless_data.CliConfig(
        level=common.GenerationLevel(level),
        data_format=common.DataFormat(data_format),
        delimiter="=",
        sgd_file=testdata_dir / "sgd_train_categorical.json",
        schema_file=testdata_dir / "sgd_train_schema.json",
        output_file=temp_output,
        randomize_items=False,
        multiple_choice=common.MultipleChoiceFormat.one_a,
    )
    slots, item_desc = create_sgd_schemaless_data.load_schema(config)
    create_sgd_schemaless_data.generate_data(config, slots, item_desc)
    assert filecmp.cmp(temp_output, ref_output)


@pytest.mark.parametrize(
    "level, data_format, data_percent, uniform_domain_distribution",
    [
        ("dst_intent", "full_desc", 0.1, True),
        ("dst_intent", "full_desc", 0.5, False),
    ],
    ids=["create_uniform_domain_10percent", "create_random_domain_50percent"],
)
def test_generate_data_sample(
    level,
    data_format,
    data_percent,
    uniform_domain_distribution,
    tmp_path,
    testdata_dir,
):
    temp_output = tmp_path / "output"
    ref_output = (
        testdata_dir
        / f"sgd_text_v2_uniform_{uniform_domain_distribution}_{data_percent}"
    )

    config = create_sgd_schemaless_data.CliConfig(
        level=common.GenerationLevel(level),
        data_format=common.DataFormat(data_format),
        delimiter="=",
        sgd_file=testdata_dir / "sgd_train.json",
        schema_file=testdata_dir / "sgd_train_schema.json",
        output_file=temp_output,
        randomize_items=False,
        data_percent=data_percent,
        uniform_domain_distribution=uniform_domain_distribution,
    )
    slots, item_desc = create_sgd_schemaless_data.load_schema(config)
    create_sgd_schemaless_data.generate_data(config, slots, item_desc)
    assert filecmp.cmp(temp_output, ref_output)


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"]))
