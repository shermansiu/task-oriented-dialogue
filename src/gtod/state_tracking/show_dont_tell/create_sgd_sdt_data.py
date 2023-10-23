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

r"""Create Show Don't Tell data from SGD Dataset.

Format: [example] <example dialogue> [slots] <slot names and values> \
[context] <current dialogue> -> [state] <dialogue state>
Example: [example] [user] can you find me a bus to lax? ... \
[slots] to_location=lax ... [context] [user] i'm looking for a bus to nyc. ... \
-> [state] to_location=nyc ...
"""

import collections
import collections.abc
import functools
import enum
import itertools
import logging
import os
import pathlib
import random
import typing as tp

import attrs
import tyro
import gtod.util
from gtod.state_tracking.show_dont_tell import common
from gtod.state_tracking.show_dont_tell import sdt_prompts
from gtod.state_tracking.show_dont_tell import sdt_utils
from gtod.state_tracking.utils import sgd_utils


Prompt = sdt_prompts.Prompt

Schemas = sgd_utils.Schemas
DialoguesDict = sgd_utils.DialoguesDict
RAND_SEED = 123
USER_SPEAKER = "USER"
SYSTEM_SPEAKER = "SYSTEM"
USER_TOK = "[user]"
SYS_TOK = "[system]"
INTENT_SLOT_VALUE_DELIMITER = "="
INPUT_TARGET_SEP = "\t"

_PROMPTS_MAP = {
    common.PromptFormat.separated: sdt_prompts.SGD_SEPARATED_ANNOTATION_PROMPTS,
}


@attrs.frozen
class Options:
    """A dataclass to store configurations for data generation."""

    sgd_dir: pathlib.Path
    sgdx_dir: pathlib.Path | None
    prompt_format: str | None
    context_format: str
    target_format: str
    lowercase: bool
    add_intents: bool
    mcq_cat_vals: bool
    mcq_intents: bool
    randomize_slots: bool
    randomize_cat_vals: bool
    randomize_intents: bool
    use_slot_ids: bool
    prompt_indices: list[int] | None


@attrs.frozen
class CreateSgdSdtConfig:
    """
    Configuration for SGD data generation.

    Args:
        input_dir: Path to SGD data directory.
        output_path: Path for the output file.
        sgdx_dir: If set, create dialogue examples using SGD-X variants from this path. e.g. /path/to/sgdx/v1/
        subdirs: Comma-separated list of dataset subdirectories to process.
        prompt_format: Format of the prompt for priming. "separated" means a dialogue followed by a separate string of slots.
        prompt_indices: Indices of the prompts for each service to be used for generating examples. Specify one or more numeric indices (starting from 0), or `None` to use all prompts for a given service.
        context_format: Format of the dialogue context.
        target_format: Format of the target. "all" and "active" respectively refer to all and only active slots being present in the target.
        add_intents: Whether to add intents.
        lowercase: Whether to lowercase the generated example.
        mcq_cat_vals: Whether to enumerate categorical values in the form of a multiple choice question in the prompt string.
        mcq_intents: Whether to enumerate intents in the form of a multiple choice question in the prompt string. Only used if `add_intents` is True.
        randomize_slots: Whether to randomize slot order of the prompt.
        randomize_cat_vals: Whether to randomize order of categorical values in prompt.
        randomize_intents: Whether to randomize order of intents in prompt. Only used if `add_intents` is True.
        use_slot_ids: Whether to use numeric slot IDs in place of slot names in the input and output strings.
        data_percent: If not 0.0, only write this proportion of data, and discard the rest of the examples. For data efficiency experiments. Not compatible with `k_shot`.
        k_shot: If not 0, sample this many examples from each service. For data efficiency experiments. Not compatible with `data_percent`.
        use_intent_slot_descs: Whether to add D3ST descriptions to prompt.

    """

    input_dir: pathlib.Path
    output_path: pathlib.Path
    sgdx_dir: pathlib.Path | None = attrs.field(default=None)
    subdirs: str = attrs.field(default="train,dev,test")
    prompt_format: common.PromptFormat = attrs.field(
        default=common.PromptFormat.separated
    )
    prompt_indices: list[int] | None = attrs.field(default=None)
    context_format: common.ContextFormat = attrs.field(
        default=common.ContextFormat.dialogue
    )
    target_format: common.TargetFormat = attrs.field(default=common.TargetFormat.all)
    add_intents: bool = attrs.field(default=False)
    lowercase: bool = attrs.field(default=True)
    mcq_cat_vals: bool = attrs.field(default=False)
    mcq_intents: bool = attrs.field(default=False)
    randomize_slots: bool = attrs.field(default=True)
    randomize_cat_vals: bool = attrs.field(default=True)
    randomize_intents: bool = attrs.field(default=True)
    use_slot_ids: bool = attrs.field(default=False)
    data_percent: float = attrs.field(default=0.0)
    k_shot: int = attrs.field(default=0)
    use_intent_slot_descs: bool = attrs.field(default=False)

    @functools.cached_property
    def subdirectories(self):
        return tuple(self.subdirs.split(","))

    @functools.cached_property
    def as_options(self) -> Options:
        return Options(
            sgd_dir=self.input_dir,
            sgdx_dir=self.sgdx_dir,
            prompt_format=self.prompt_format,
            context_format=self.context_format,
            target_format=self.target_format,
            add_intents=self.add_intents,
            mcq_cat_vals=self.mcq_cat_vals,
            mcq_intents=self.mcq_intents,
            randomize_slots=self.randomize_slots,
            randomize_cat_vals=self.randomize_cat_vals,
            randomize_intents=self.randomize_intents,
            lowercase=self.lowercase,
            use_slot_ids=self.use_slot_ids,
            prompt_indices=self.prompt_indices,
        )


config: CreateSgdSdtConfig | None = None


@attrs.frozen
class Example:
    """Dataclass for single SDT example.

    Attributes:
        example_str: The example string.
        services: The services this example belongs to.
    """

    example_str: str
    services: list[str]


def _generate_utt_str(utterance: str, speaker: str) -> str:
    """Generates the utterance string for an example."""
    if speaker == USER_SPEAKER:
        prefix = USER_TOK
    elif speaker == SYSTEM_SPEAKER:
        prefix = SYS_TOK
    else:
        raise ValueError(
            f"Speaker must be one of {USER_SPEAKER} "
            f"or {SYSTEM_SPEAKER}. Found {speaker}"
        )

    # Occasionally some examples include newlines in the middle
    utterance = utterance.replace("\n", " ")

    return " ".join([prefix, utterance])


def build_example(
    input_strs: collections.abc.Sequence[str],
    target_str: str,
    additional_strs: collections.abc.Sequence[str],
    services: collections.abc.Sequence[str],
    lowercase: bool,
) -> Example:
    """Builds a single example in TSV format."""
    example_str = " ".join(input_strs) + INPUT_TARGET_SEP + target_str
    if additional_strs:
        example_str += INPUT_TARGET_SEP + INPUT_TARGET_SEP.join(additional_strs)

    if lowercase:
        example_str = example_str.lower()

    return Example(example_str=example_str.strip(), services=list(services))


def create_examples_from_dialogue(
    dialogue: collections.abc.Mapping[str, tp.Any],
    service_to_prompts: dict[str, list[Prompt]] | None,
    service_to_schema: collections.abc.Mapping[str, sgd_utils.Schema],
    options: Options,
) -> list[Example]:
    """Returns example strings created from a dialogue.

    Args:
        dialogue: A single dialogue containing multiple turns and frames
        service_to_prompts: A map from SGD service to a list of prompts
        service_to_schema: A map from SGD service to schema
        options: An object containing various options related to example generation
    """
    utt_strs = []
    example_strs = []

    for turn_idx, turn in enumerate(dialogue["turns"]):
        # Format utterances
        utt_strs.append(
            _generate_utt_str(utterance=turn["utterance"], speaker=turn["speaker"])
        )

        # Don't create examples out of system turns for DST
        if turn["speaker"] != USER_SPEAKER:
            continue

        for frame_idx, frame in enumerate(turn["frames"]):
            # Create prompt
            (
                prompt_str,
                ordered_slots,
                slot_to_cat_val_to_id,
                intent_to_id,
            ) = sdt_utils.generate_prompt_str(
                keys=[frame["service"]],
                key_to_prompts=service_to_prompts,
                prompt_indices=options.prompt_indices,
                add_intents=options.add_intents,
                mcq_cat_vals=options.mcq_cat_vals,
                mcq_intents=options.mcq_intents,
                randomize_slots=options.randomize_slots,
                randomize_cat_vals=options.randomize_cat_vals,
                randomize_intents=options.randomize_intents,
                use_slot_ids=options.use_slot_ids,
                key_to_schema=service_to_schema,
            )

            # Create context
            context_str = sdt_utils.generate_context_str(
                utt_strs, options.context_format
            )

            # Create target
            target_str = sdt_utils.generate_target_str(
                dialogue_state=frame["state"]["slot_values"],
                active_intent=frame["state"]["active_intent"],
                add_intents=options.add_intents,
                ordered_slots=ordered_slots,
                slot_to_cat_val_to_id=slot_to_cat_val_to_id,
                intent_to_id=intent_to_id,
                target_format=options.target_format,
                use_slot_ids=options.use_slot_ids,
            )

            example_strs.append(
                build_example(
                    input_strs=[prompt_str, context_str],
                    target_str=target_str,
                    additional_strs=[
                        dialogue["dialogue_id"],
                        str(turn_idx),
                        str(frame_idx),
                    ],
                    services=dialogue["services"],
                    lowercase=options.lowercase,
                )
            )

    return example_strs


def main() -> None:
    if config.data_percent > 0.0 and config.k_shot > 0:
        raise ValueError("Only one of data_percent and k_shot can be specified!")

    # Set random seed
    random.seed(RAND_SEED)

    options = config.as_options

    # Load dataset - SGD-X if provided, otherwise SGD
    sgd_data_dir = options.sgdx_dir or options.sgd_dir
    subdir_to_schema, subdir_to_dialogues = sgd_utils.load_dataset(
        data_dir=sgd_data_dir, subdirs=config.subdirectories
    )

    # If enabled, create map from service to schema for adding D3ST descriptions
    if config.use_intent_slot_descs.value:
        service_to_schema = sgd_utils.dedupe_and_unnest_schemas(subdir_to_schema)
    else:
        service_to_schema = False

    # Fetch prompts and replace with SGD-X version if applicable
    if options.prompt_format:
        service_to_prompts = _PROMPTS_MAP[options.prompt_format]
        if options.sgdx_dir:
            service_to_prompts = sdt_utils.create_sgdx_prompts(
                service_to_prompts, options.sgd_dir, options.sgdx_dir
            )
    else:
        service_to_prompts = None

    # Create output directory if needed
    config.output_path.parent.mkdir(exist_ok=True, parents=True)

    # Loop through dialogues and create examples
    with config.output_path.open("w") as outfile:
        examples = []
        for subdir, dfile_to_dialogues in subdir_to_dialogues.items():
            logging.info("Processing subdir %s", subdir)

            for dfile, dialogues in dfile_to_dialogues.items():
                logging.info("Processing file %s", dfile)

                for dialogue in dialogues:
                    examples.extend(
                        create_examples_from_dialogue(
                            dialogue, service_to_prompts, service_to_schema, options
                        )
                    )

        # Optionally sample a proportion of examples only
        if config.data_percent.value > 0.0:
            examples = random.sample(
                examples, int(config.data_percent.value * len(examples))
            )
        elif config.k_shot.value > 0:
            # A dict of service to a list of examples belonging to that service.
            service_to_examples = collections.defaultdict(list)
            for example in examples:
                for service in example.services:
                    service_to_examples[service].append(example)

            # Sample K examples from each service
            examples = itertools.chain(
                *[
                    random.sample(examples_by_service, k=config.k_shot)
                    for examples_by_service in service_to_examples.values()
                ]
            )

        # Write example strings to file
        for e in examples:
            outfile.write(f"{e.example_str}\n")


if __name__ == "__main__":
    config = tyro.cli(CreateSgdSdtConfig)
    main()
