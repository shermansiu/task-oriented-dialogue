# Copyright 2021 Google Research.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute slot error rate (SER) metric."""

import csv
import collections
import json
import pathlib

import attrs
import cattrs
import pandas as pd
import tyro

import gtod


@attrs.config
class SerConfig:
    """SER Config.

    Attributes:
        predictions_path (str): Path to T5 predictions file.
        inputs_path (str): Path to tsv dataset file.
        data_dir (str): Path to SGD dataset. The directory should include subdirectories for train, dev, and test, each with its own schema.json file.
    """

    predictions_path: pathlib.Path
    inputs_path: pathlib.Path
    data_dir: pathlib.Path


def get_ser_slots(data_dir: pathlib.Path) -> collections.abc.Mapping[str, str]:
    """Get a mapping of service names to non-categorical slot names."""
    schemas = []
    for split in list(gtod.util.DatasetSplitConfig):
        with data_dir.joinpath(f"{split}/schema.json") as f:
            schemas += cattrs.structure(json.load(f), Service)
    permissible_slots = collections.defaultdict(list)
    for schema in schemas:
        for slot in schema.slots:
            if not slot.is_categorical:
                # SER is claculated only for categorical slots.
                permissible_slots[schema.service_name].append(slot.name)
        return permissible_slots


def example_ser(mr: Dialogue, prediction: str, permissible_slots) -> bool:
    """Calculates slot error rate for a single prediction."""
    prediction = prediction.lower()
    for frame in mr.frames:
        service = frame.service
        for action in frame.actions:
            slot = action["slot"]
            if slot not in permissible_slots[service]:
                continue
            values = action["values"]
            for value in values:
                value = value.lower()
                if value not in prediction:
                    return False
    return True


def calculate_ser(data, permissible_slots):
    """Calculates slot error rate for a set of predictions."""
    df = pd.DataFrame(data)
    df["is_wrong"] = df.apply(
        lambda x: not example_ser(x["mr"], x["prediction"], permissible_slots), axis=1
    )
    results = {}
    results["overall"] = df["is_wrong"].mean()
    df_ser = (
        df.groupby("tag").apply(lambda x: x["is_wrong"].mean()).reset_index(name="ser")
    )
    results.update(dict(df_ser.values))
    results = {k: v * 100 for k, v in results.items()}
    return results


def prepare_data(inputs_path, predictions_path):
    """Prepare inputs and predictions for slot error rate calculation."""
    unseen_domains = ["Alarm", "Messaging", "Payment", "Train"]
    predictions = [line.strip("\n") for line in tf.io.gfile.GFile(predictions_path)]
    reader = csv.reader(tf.io.gfile.GFile(inputs_path), delimiter="\t")
    data = list(reader)
    mrs = [json.loads(mr) for _, _, mr in data]
    inputs = [inp for inp, _, _ in data]
    targets = [target for _, target, _ in data]
    services = [mr["frames"][0]["service"] for mr in mrs]
    domains = [service.split("_")[0] for service in services]
    tags = ["unseen" if domain in unseen_domains else "seen" for domain in domains]
    data = {
        "mr": mrs,
        "input": inputs,
        "prediction": predictions,
        "target": targets,
        "tag": tags,
    }
    return data


def main(_):
    data = prepare_data(FLAGS.inputs_path, FLAGS.predictions_path)
    permissible_slots = get_ser_slots(FLAGS.data_dir)
    ser_results = calculate_ser(data, permissible_slots)
    print(ser_results)


if __name__ == "__main__":
    app.run(main)
