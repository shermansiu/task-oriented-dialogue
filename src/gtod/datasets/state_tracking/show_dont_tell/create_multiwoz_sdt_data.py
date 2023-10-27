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

r"""Create Show Don't Tell data from Multiwoz Dataset.

Format: [example] <example dialogue> [slots] <slot names and values> \
[context] <current dialogue> -> [state] <dialogue state>
e.g. `[example] [user] can you find me a train to lax? ... \
[slots] train-destination=lax ... [context] [user] i'm looking for a train
to nyc. ... \ -> [state] train-destination=nyc ...`
"""

import collections
import functools
import pathlib
import random

import attrs
import tyro

from gtod.datasets.state_tracking.show_dont_tell import sdt_utils, common, sdt_prompts
from gtod.datasets.state_tracking.utils import text_to_text_utils, multiwoz_utils


@attrs.frozen
class CliConfig:
    """Configuration for creating MultiWoZ SDT data.

    Attributes:
        input_dir: Path to the original MultiWOZ datasets.
        output_dir: Output file path.
        schema_file: MultiWOZ schema file in 2.2/SGD format.
        multiwoz_version: MultiWOZ dataset version. One of ["2.1", "2.2", "2.3", "2.4"].
        is_trade: Whether the data is TRADE-preprocessed or not.
        prompt_format: Format of the prompt for priming. Only "separated" is supported.
            "separated" means a dialogue followed by a separate string of slots.
        prompt_indices: Indices of the prompts for each service to be used for generating examples.
            Specify one or more numeric indices (starting from 0), or `None` to use all prompts for a given service.
        context_format: Format of the dialogue context. Only "dialogue" is supported.
        target_format: Format of the target. "all" refers to all slots being in the target.
        lowercase: Whether to lowercase the generated example.
        mcq_cat_vals: Whether to enumerate categorical values in the form of a multiple choice question in the prompt.
        randomize_slots: Whether to randomize slot order of the prompt.
        randomize_cat_vals: Whether to randomize the order of categorical values in the prompt.
        shuffle: Whether to randomly shuffle examples before writing out.
        use_active_domains_only: If true, only include domains that are active in this dialogue in the prompt.
        blocked_domains: Domains to exclude. Used for zero-shot cross-domain experiments.
    """

    input_dir: pathlib.Path
    output_dir: pathlib.Path
    schema_file: pathlib.Path
    multiwoz_version: multiwoz_utils.MultiwozVersion = attrs.field(
        default=multiwoz_utils.MultiwozVersion.v21
    )
    is_trade: bool = attrs.field(default=True)
    prompt_format: common.PromptFormat = attrs.field(
        default=common.PromptFormat.separated
    )
    prompt_indices: list[int] | None = attrs.field(default=None)
    context_format: common.ContextFormat = attrs.field(
        default=common.ContextFormat.dialogue
    )
    target_format: common.TargetFormat = attrs.field(default=common.TargetFormat.all)
    lowercase: bool = attrs.field(default=True)
    mcq_cat_vals: bool = attrs.field(default=False)
    randomize_slots: bool = attrs.field(default=True)
    randomize_cat_vals: bool = attrs.field(default=True)
    shuffle: bool = attrs.field(default=True)
    use_active_domains_only: bool = attrs.field(default=False)
    blocked_domains: list[str] = attrs.field(factory=list)

    @functools.cached_property
    def as_options(self):
        return Options(
            multiwoz_version=self.multiwoz_version,
            is_trade=self.is_trade,
            prompt_format=self.prompt_format,
            prompt_indices=self.prompt_indices,
            context_format=self.context_format,
            target_format=self.target_format,
            mcq_cat_vals=self.mcq_cat_vals,
            randomize_slots=self.randomize_slots,
            randomize_cat_vals=self.randomize_cat_vals,
            use_active_domains_only=self.use_active_domains_only,
            blocked_domains=set(self.blocked_domains),
            lowercase=self.lowercase,
        )


# Use Ordereddict for JSON to preserve field order.
Json = collections.OrderedDict
MultiwozData = multiwoz_utils.MultiwozData
SchemaInfo = multiwoz_utils.SchemaInfo
TextToTextExample = text_to_text_utils.TextToTextExample
MULTIWOZ_DOMAINS = [
    "attraction",
    "bus",
    "hospital",
    "hotel",
    "restaurant",
    "taxi",
    "train",
]
USER_TOK = "[user]"
SYS_TOK = "[system]"

_PROMPTS_MAP = {
    common.PromptFormat.separated: sdt_prompts.MW_SEPARATED_ANNOTATION_PROMPTS,
}


@attrs.frozen
class Options:
    """Options for generating SDT examples."""

    multiwoz_version: multiwoz_utils.MultiwozVersion
    is_trade: bool
    prompt_format: common.PromptFormat
    prompt_indices: list[int] | None
    context_format: common.ContextFormat
    target_format: common.TargetFormat
    mcq_cat_vals: bool
    randomize_slots: bool
    randomize_cat_vals: bool
    use_active_domains_only: bool
    blocked_domains: set[str]
    lowercase: bool


def _normalize_multiwoz_slot_values(
    dialogue_state: dict[str, str], multiwoz_version: str
) -> dict[str, list[str]]:
    """Normalizes multiwoz slot values into a common format."""
    new_state = {}

    for slot_name, values in dialogue_state.items():
        if "|" in values:
            values = values.split("|")
        elif ">" in values:
            values = values.split(">")
        elif "<" in values:
            values = values.split("<")
        elif multiwoz_version != "2.2":
            # Put values into a list to accommodate 2.2 format giving multiple values
            values = [values]

        if not isinstance(values, list):
            raise ValueError(
                '"values" for a slot must be of list type. Actual: '
                f"{type(values)}. values: {values}"
            )

        new_state[slot_name] = values

    return new_state


def _process_one_turn(
    dialog_id: str,
    turn: int,
    belief_state: dict[str, str],
    history_utterances: list[str],
    options: Options,
) -> TextToTextExample:
    """Processes a single dialogue turn into a `TextToTextExample`."""
    # Fetch prompts
    domain_to_prompts = (
        _PROMPTS_MAP[options.prompt_format] if options.prompt_format else None
    )

    # Create prompt
    if options.use_active_domains_only:
        domains = list(multiwoz_utils.extract_domains(belief_state))
    else:
        domains = MULTIWOZ_DOMAINS
    (
        prompt_str,
        ordered_slots,
        ordered_slot_to_cat_val_to_id,
        intent_to_id,
    ) = sdt_utils.generate_prompt_str(
        keys=sorted(domains),
        key_to_prompts=domain_to_prompts,
        prompt_indices=options.prompt_indices,
        add_intents=False,
        mcq_cat_vals=options.mcq_cat_vals,
        mcq_intents=False,
        randomize_slots=options.randomize_slots,
        randomize_intents=False,
        randomize_cat_vals=options.randomize_cat_vals,
    )

    # Create context
    context_str = sdt_utils.generate_context_str(
        history_utterances, options.context_format
    )

    # Create target
    norm_dialogue_state = _normalize_multiwoz_slot_values(
        belief_state, options.multiwoz_version
    )
    # MultiWoZ2.1 does not have active intents, hence setting to empty string.
    target_str = sdt_utils.generate_target_str(
        dialogue_state=norm_dialogue_state,
        active_intent="",
        add_intents=False,
        ordered_slots=ordered_slots,
        slot_to_cat_val_to_id=ordered_slot_to_cat_val_to_id,
        intent_to_id=intent_to_id,
        target_format=options.target_format,
        use_slot_ids=False,
    )

    # Lowercase
    if options.lowercase:
        prompt_str = prompt_str.lower()
        context_str = context_str.lower()
        target_str = target_str.lower()

    return TextToTextExample(
        src=" ".join([prompt_str, context_str.strip()]).strip(),
        tgt=target_str,
        dialog_id=dialog_id,
        turn=turn,
    )


def create_sdt_examples(json_data: Json, options: Options) -> list[TextToTextExample]:
    """Converts raw MultiWOZ data into "Show Don't Tell" examples."""
    examples = []

    for dialog_id, dialog_json in json_data.items():
        history_utterances = []

        dialog_key = "dialogue" if options.is_trade else "log"
        belief_state_key = "belief_state" if options.is_trade else "metadata"
        for turn, utterance_json in enumerate(dialog_json[dialog_key]):
            # Process utterance json
            if options.is_trade:
                sys_utt = utterance_json["system_transcript"].strip().replace("\t", " ")
                user_utt = utterance_json["transcript"].strip().replace("\t", " ")
                if turn == 0:
                    history_utterances.append(f"[user] {user_utt}")
                else:
                    history_utterances.append(f"[system] {sys_utt} [user] {user_utt}")
                is_system = True
            else:
                is_system = turn % 2 == 1

            belief_state = multiwoz_utils.extract_belief_state(
                metadata_json=utterance_json[belief_state_key],
                is_trade=options.is_trade,
            )

            # State, action, and response only in system turns for non-TRADE data
            if is_system:
                # Skip turns if a blocked domain appears (including adding to history)
                domains_in_turn = multiwoz_utils.extract_domains(belief_state)
                if options.blocked_domains & domains_in_turn:
                    continue
                examples.append(
                    _process_one_turn(
                        dialog_id, turn, belief_state, history_utterances, options
                    )
                )

            # Update history_utterances for non-TRADE data
            if not options.is_trade:
                utterance = (
                    utterance_json["text"].strip().replace("\t", " ").replace("\n", " ")
                )
                history_utterances.append(
                    f"{SYS_TOK if is_system else USER_TOK} " f"{utterance}"
                )

    return examples


def main(config: CliConfig):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=config.input_dir,
        multiwoz_version=config.multiwoz_version,
        is_trade=config.is_trade,
    )

    options = config.as_options

    # Create SDT examples
    split_to_examples = {
        "train": create_sdt_examples(multiwoz_data.train_json, options),
        "dev": create_sdt_examples(multiwoz_data.dev_json, options),
        "test": create_sdt_examples(multiwoz_data.test_json, options),
    }

    # Write out examples
    if config.shuffle:
        for examples in split_to_examples.values():
            random.shuffle(examples)
    split_to_examples["dev_test"] = split_to_examples["dev"] + split_to_examples["test"]
    for split, examples in split_to_examples.items():
        text_to_text_utils.write_data(examples, config.output_dir / f"{split}")


if __name__ == "__main__":
    config = tyro.cli(CliConfig)
    main(config)
