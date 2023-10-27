# Copyright 2021 Google Research.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for creating text-to-text data."""

import collections
import collections.abc
import logging
import pathlib
import attrs
import datasets


# TODO(jeffreyzhao): Support extending with multiple fields
@attrs.frozen
class TextToTextExample:
    """A single text-to-text dialogue example.

    Attributes:
        src: Input text for the model.
        tgt: Target text for the model.
        dialog_id: Id of dialog this example was generated from.
        turn: Turn of dialog this example was generated from.
        metadata: Any other key-value pairs to be included in the output TF Example.
        frame: Frame of the dialog this example was generated from.
    """

    src: str
    tgt: str
    dialog_id: str
    turn: int
    metadata: dict[str, str] = attrs.field(factory=dict)
    frame: int = 0


def write_data(
    examples: collections.abc.MutableSequence[TextToTextExample],
    output_path: pathlib.Path,
) -> None:
    """Writes examples to the given output path.

    Args:
        examples: A list of formatted examples to write out
        output_path: The file path to write examples out to
    """

    output_path.parent.mkdir(exist_ok=True, parents=True)

    input = []
    value = []
    dialog_id = []
    turn = []
    dataset_dict = collections.defaultdict(list)

    for example in examples:
        input.append(example.src)
        value.append(example.tgt)
        dialog_id.append(example.dialog_id)
        turn.append(example.turn)

        for key in examples[0].metadata:
            assert key not in ("input", "value", "dialog_id", "turn")
            dataset_dict[key].append(example.metadata.get(key, ""))

    dataset = datasets.Dataset.from_dict(dataset_dict)
    dataset.save_to_disk(output_path)

    logging.info("Wrote %s with %d examples", output_path.name, len(examples))
