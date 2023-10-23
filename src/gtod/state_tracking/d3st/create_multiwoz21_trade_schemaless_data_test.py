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

"""Tests for create_multiwoz21_trade_schemaless_data."""

import random
import shutil
import sys

import pytest

from gtod.state_tracking.d3st import create_multiwoz21_trade_schemaless_data
from gtod.state_tracking.d3st import common
from gtod.state_tracking.utils import multiwoz_utils

TEST_DIR = "zero_shot_task_oriented_dialog/testdata"


@pytest.fixture
def schema_file(tmp_path, testdata_dir):
    random.seed(31415)
    schema_file_ = tmp_path / "schema.json"

    shutil.copy(
        testdata_dir / "multiwoz_data_trade.json",
        tmp_path / "train_dials.json",
    )
    shutil.copy(
        testdata_dir / "multiwoz_slot_descriptions.json",
        tmp_path / "slot_descriptions.json",
    )
    shutil.copy(
        testdata_dir / "multiwoz_schema_schemaless.json",
        schema_file_,
    )

    # Touch empty files for (val|test)ListFile.json
    (tmp_path / "dev_dials.json").touch()
    (tmp_path / "test_dials.json").touch()

    return schema_file_


def test_item_name(schema_file, tmp_path):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=tmp_path, multiwoz_version="2.1", is_trade=True
    )
    schema_info = multiwoz_utils.load_schema(schema_file)
    examples = create_multiwoz21_trade_schemaless_data.create_schemaless_data(
        multiwoz_data.train_json,
        schema_info,
        multiwoz_data.slot_descriptions,
        create_multiwoz21_trade_schemaless_data.Options(
            description_type=common.DescriptionType("item_name"),
            delimiter=":",
            multiple_choice=common.MultipleChoiceFormat("none"),
            use_active_domains_only=False,
            blocked_domains=set(),
        ),
    )

    assert len(examples) == 7
    assert examples[1].src == (
        "0:restaurant-area 1:bus-destination 2:train-departure "
        "3:restaurant-name 4:attraction-name 5:bus-leaveat 6:hotel-name "
        "7:restaurant-time 8:train-people 9:restaurant-food 10:taxi-departure "
        "11:bus-day 12:train-day 13:hotel-type 14:train-leaveat "
        "15:taxi-arriveby 16:hotel-stay 17:train-arriveby 18:taxi-leaveat "
        "19:hotel-stars 20:hotel-people 21:hotel-pricerange 22:hotel-parking "
        "23:hotel-internet 24:restaurant-day 25:attraction-type "
        "26:restaurant-pricerange 27:hotel-area 28:train-destination "
        "29:bus-departure 30:hospital-department 31:hotel-day "
        "32:attraction-area 33:restaurant-people 34:taxi-destination [user] "
        "hi , can you help me find a train departing from cambridge ? [system] "
        "what is your destination ? [user] i am going to leicester"
    )
    assert examples[1].tgt == "[states] 2:cambridge 28:leicester [intents] [req_slots]"
    assert examples[1].dialog_id == "mul0708.json"
    assert examples[1].turn == 1


def test_shuffled_item_name(schema_file, tmp_path):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=tmp_path, multiwoz_version="2.1", is_trade=True
    )
    schema_info = multiwoz_utils.load_schema(schema_file)
    examples = create_multiwoz21_trade_schemaless_data.create_schemaless_data(
        multiwoz_data.train_json,
        schema_info,
        multiwoz_data.slot_descriptions,
        create_multiwoz21_trade_schemaless_data.Options(
            description_type=common.DescriptionType("shuffled_item_name"),
            delimiter=":",
            multiple_choice=common.MultipleChoiceFormat.one_a,
            use_active_domains_only=False,
            blocked_domains=set(),
        ),
    )

    assert len(examples) == 7
    assert examples[1].src == (
        "0:eharnoileetg-pcr 0a) peixnsvee 0b) hacpe 0c) aoedetmr "
        "1:lteaiarnea-tv 2:onatitmtercaan- 3:eep-oopethll 4:aanenrtsuamt-re "
        "5:arrity-invabre 6:oseraiint-nnatidt 7:al-eortaeh 7a) tonrh 7b) twes "
        "7c) atse 7d) sutho 7e) enetrc 8:eo-hnntlerteit 8a) on 8b) esy "
        "9:hr-aotpkneigl 9a) on 9b) reef 9c) sey 10:eaeraanurr-atts 10a) tesa "
        "10b) tnrece 10c) twse 10d) nhort 10e) tsouh 11:meehproiapsndltatt- "
        "12:basu-dy 12a) aundsy 12b) tsdayaur 12c) easuytd 12d) rdyiaf 12e) "
        "edsyedawn 12f) hrdtayus 12g) ynoadm 13:sert-sthalo 14:tloeehpy-t 14a) "
        "thusguseeo 14b) ethol 15:alitv-xeetaa 16:aptttaeyio-nrtc "
        "17:-eropletanresputa 18:-anmtesuetartri 19:e-penpartilo "
        "20:uabdpturse-er 21:teoyhlta-s 22:uaas-petreernrcairtng 22a) "
        "expivesen 22b) echap 22c) artmeode 23:aotmneh-el 24:darrf-nesoaottu "
        "25:uteavabel-s 26:oaldye-th 26a) aidfyr 26b) ysuadn 26c) sadneeydw "
        "26d) stduyhra 26e) sduytae 26f) utdaasry 26g) oanmyd 27:adt-inary "
        "27a) sueydat 27b) wysnededa 27c) aonmyd 27d) syuadn 27e) uraasdty "
        "27f) ayrdfi 27g) tuydhsra 28:uearrtayastdn- 28a) oymnda 28b) irdafy "
        "28c) dayrauts 28d) dsyadnwee 28e) tdsuyhar 28f) ayunds 28g) tuyeads "
        "29:natei-rttacaaro 29a) sewt 29b) stae 29c) rhotn 29d) uosth 29e) "
        "crneet 29f) bracgmdei 30:xiiettsatn-iando 31:obasi-eitsdutnn "
        "32:arbviyr-itaex 33:pdtnreeuiaartr- 34:d-turaeerapxit [user] hi , can "
        "you help me find a train departing from cambridge ? [system] what is "
        "your destination ? [user] i am going to leicester"
    )
    assert examples[1].tgt == "[states] 6:leicester 33:cambridge [intents] [req_slots]"
    assert examples[1].dialog_id == "mul0708.json"
    assert examples[1].turn == 1


def test_full_desc(schema_file, tmp_path):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=tmp_path, multiwoz_version="2.1", is_trade=True
    )
    schema_info = multiwoz_utils.load_schema(schema_file)
    examples = create_multiwoz21_trade_schemaless_data.create_schemaless_data(
        multiwoz_data.train_json,
        schema_info,
        multiwoz_data.slot_descriptions,
        create_multiwoz21_trade_schemaless_data.Options(
            description_type=common.DescriptionType("full_desc"),
            delimiter=":",
            multiple_choice=common.MultipleChoiceFormat("none"),
            use_active_domains_only=False,
            blocked_domains=set(),
        ),
    )

    assert len(examples) == 7
    assert examples[1].src == (
        "0:area or place of the restaurant 1:destination of bus 2:departure "
        "location of the train 3:name of the restaurant 4:name of the "
        "attraction 5:leaving time of bus 6:name of the hotel 7:time of the "
        "restaurant booking 8:number of people booking for train 9:food type "
        "for the restaurant 10:departure location of taxi 11:day to use the "
        "bus tickets 12:day of the train 13:what is the type of the hotel "
        "14:leaving time for the train 15:arrival time of taxi 16:length of "
        "stay at the hotel 17:arrival time of the train 18:leaving time of "
        "taxi 19:star rating of the hotel 20:number of people for the hotel "
        "booking 21:price budget of the hotel 22:parking facility at the hotel "
        "23:internet option at the hotel 24:day of the restaurant booking "
        "25:type of the attraction 26:price budget for the restaurant 27:area "
        "or place of the hotel 28:destination of the train 29:departure "
        "location of bus 30:name of hospital department 31:day of the hotel "
        "booking 32:area or place of the attraction 33:number of people "
        "booking the restaurant 34:destination of taxi [user] hi , can you "
        "help me find a train departing from cambridge ? [system] what is your "
        "destination ? [user] i am going to leicester"
    )
    assert examples[1].tgt == "[states] 2:cambridge 28:leicester [intents] [req_slots]"
    assert examples[1].dialog_id == "mul0708.json"
    assert examples[1].turn == 1


def test_full_desc_with_domain(schema_file, tmp_path):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=tmp_path, multiwoz_version="2.1", is_trade=True
    )
    schema_info = multiwoz_utils.load_schema(schema_file)
    examples = create_multiwoz21_trade_schemaless_data.create_schemaless_data(
        multiwoz_data.train_json,
        schema_info,
        multiwoz_data.slot_descriptions,
        create_multiwoz21_trade_schemaless_data.Options(
            description_type=common.DescriptionType("full_desc_with_domain"),
            delimiter=":",
            multiple_choice=common.MultipleChoiceFormat("none"),
            use_active_domains_only=False,
            blocked_domains=set(),
        ),
    )

    assert len(examples) == 7
    assert examples[1].src == (
        "0:restaurant-area or place of the restaurant 1:bus-destination of "
        "bus 2:train-departure location of the train 3:restaurant-name of the "
        "restaurant 4:attraction-name of the attraction 5:bus-leaving time of "
        "bus 6:hotel-name of the hotel 7:restaurant-time of the restaurant "
        "booking 8:train-number of people booking for train 9:restaurant-food "
        "type for the restaurant 10:taxi-departure location of taxi 11:bus-day "
        "to use the bus tickets 12:train-day of the train 13:hotel-what is the "
        "type of the hotel 14:train-leaving time for the train 15:taxi-arrival "
        "time of taxi 16:hotel-length of stay at the hotel 17:train-arrival "
        "time of the train 18:taxi-leaving time of taxi 19:hotel-star rating "
        "of the hotel 20:hotel-number of people for the hotel booking "
        "21:hotel-price budget of the hotel 22:hotel-parking facility at the "
        "hotel 23:hotel-internet option at the hotel 24:restaurant-day of the "
        "restaurant booking 25:attraction-type of the attraction "
        "26:restaurant-price budget for the restaurant 27:hotel-area or place "
        "of the hotel 28:train-destination of the train 29:bus-departure "
        "location of bus 30:hospital-name of hospital department 31:hotel-day "
        "of the hotel booking 32:attraction-area or place of the attraction "
        "33:restaurant-number of people booking the restaurant "
        "34:taxi-destination of taxi [user] hi , can you help me find a train "
        "departing from cambridge ? [system] what is your destination ? [user] "
        "i am going to leicester"
    )
    assert examples[1].tgt == "[states] 2:cambridge 28:leicester [intents] [req_slots]"
    assert examples[1].dialog_id == "mul0708.json"
    assert examples[1].turn == 1


def test_delimiter(schema_file, tmp_path):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=tmp_path, multiwoz_version="2.1", is_trade=True
    )
    schema_info = multiwoz_utils.load_schema(schema_file)
    examples = create_multiwoz21_trade_schemaless_data.create_schemaless_data(
        multiwoz_data.train_json,
        schema_info,
        multiwoz_data.slot_descriptions,
        create_multiwoz21_trade_schemaless_data.Options(
            description_type=common.DescriptionType("full_desc"),
            delimiter="=",
            multiple_choice=common.MultipleChoiceFormat("none"),
            use_active_domains_only=False,
            blocked_domains=set(),
        ),
    )

    assert len(examples) == 7
    assert examples[1].src == (
        "0=area or place of the restaurant 1=destination of bus 2=departure "
        "location of the train 3=name of the restaurant 4=name of the "
        "attraction 5=leaving time of bus 6=name of the hotel 7=time of the "
        "restaurant booking 8=number of people booking for train 9=food type "
        "for the restaurant 10=departure location of taxi 11=day to use the "
        "bus tickets 12=day of the train 13=what is the type of the hotel "
        "14=leaving time for the train 15=arrival time of taxi 16=length of "
        "stay at the hotel 17=arrival time of the train 18=leaving time of "
        "taxi 19=star rating of the hotel 20=number of people for the hotel "
        "booking 21=price budget of the hotel 22=parking facility at the hotel "
        "23=internet option at the hotel 24=day of the restaurant booking "
        "25=type of the attraction 26=price budget for the restaurant 27=area "
        "or place of the hotel 28=destination of the train 29=departure "
        "location of bus 30=name of hospital department 31=day of the hotel "
        "booking 32=area or place of the attraction 33=number of people "
        "booking the restaurant 34=destination of taxi [user] hi , can you "
        "help me find a train departing from cambridge ? [system] what is your "
        "destination ? [user] i am going to leicester"
    )
    assert examples[1].tgt == "[states] 2=cambridge 28=leicester [intents] [req_slots]"
    assert examples[1].dialog_id == "mul0708.json"
    assert examples[1].turn == 1


def test_multiple_choice_a(schema_file, tmp_path):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=tmp_path, multiwoz_version="2.1", is_trade=True
    )
    schema_info = multiwoz_utils.load_schema(schema_file)

    # The testdata example doesn't have any categorical slots, so mock some.
    schema_info.slots_by_domain["train"]["train-departure"] = multiwoz_utils.SlotInfo(
        is_categorical=True, possible_values=["leicester", "cambridge"]
    )
    schema_info.slots_by_domain["train"]["train-destination"] = multiwoz_utils.SlotInfo(
        is_categorical=True, possible_values=["leicester", "cambridge"]
    )

    examples = create_multiwoz21_trade_schemaless_data.create_schemaless_data(
        multiwoz_data.train_json,
        schema_info,
        multiwoz_data.slot_descriptions,
        create_multiwoz21_trade_schemaless_data.Options(
            description_type=common.DescriptionType("full_desc"),
            delimiter=":",
            multiple_choice=common.MultipleChoiceFormat("a"),
            use_active_domains_only=True,
            blocked_domains=set(),
        ),
    )

    assert len(examples) == 7
    assert examples[1].src == (
        "0:departure location of the train a) leicester b) cambridge 1:arrival "
        "time of the train 2:number of people booking for train 3:leaving time "
        "for the train 4:day of the train a) sunday b) friday c) tuesday "
        "d) monday e) wednesday f) saturday g) thursday 5:destination of the "
        "train a) leicester b) cambridge [user] hi , can you help me find a "
        "train departing from cambridge ? [system] what is your destination ? "
        "[user] i am going to leicester"
    )
    assert examples[1].tgt == "[states] 0:b 5:a [intents] [req_slots]"
    assert examples[1].dialog_id == "mul0708.json"
    assert examples[1].turn == 1


def test_multiple_choice_1a(schema_file, tmp_path):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=tmp_path, multiwoz_version="2.1", is_trade=True
    )
    schema_info = multiwoz_utils.load_schema(schema_file)

    # The testdata example doesn't have any categorical slots, so mock some.
    schema_info.slots_by_domain["train"]["train-departure"] = multiwoz_utils.SlotInfo(
        is_categorical=True, possible_values=["leicester", "cambridge"]
    )
    schema_info.slots_by_domain["train"]["train-destination"] = multiwoz_utils.SlotInfo(
        is_categorical=True, possible_values=["leicester", "cambridge"]
    )

    examples = create_multiwoz21_trade_schemaless_data.create_schemaless_data(
        multiwoz_data.train_json,
        schema_info,
        multiwoz_data.slot_descriptions,
        create_multiwoz21_trade_schemaless_data.Options(
            description_type=common.DescriptionType("full_desc"),
            delimiter=":",
            multiple_choice="1a",
            use_active_domains_only=True,
            blocked_domains=set(),
        ),
    )

    assert len(examples) == 7
    assert examples[1].src == (
        "0:departure location of the train 0a) leicester 0b) cambridge "
        "1:arrival time of the train 2:number of people booking for train "
        "3:leaving time for the train 4:day of the train 4a) sunday 4b) friday "
        "4c) tuesday 4d) monday 4e) wednesday 4f) saturday 4g) thursday "
        "5:destination of the train 5a) leicester 5b) cambridge [user] hi , "
        "can you help me find a train departing from cambridge ? [system] what "
        "is your destination ? [user] i am going to leicester"
    )
    assert examples[1].tgt == "[states] 0:0b 5:5a [intents] [req_slots]"
    assert examples[1].dialog_id == "mul0708.json"
    assert examples[1].turn == 1


def test_blocked_one_domain(schema_file, tmp_path):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=tmp_path, multiwoz_version="2.1", is_trade=True
    )
    schema_info = multiwoz_utils.load_schema(schema_file)

    # The testdata example doesn't have any categorical slots, so mock some.
    schema_info.slots_by_domain["train"]["train-departure"] = multiwoz_utils.SlotInfo(
        is_categorical=True, possible_values=["leicester", "cambridge"]
    )
    schema_info.slots_by_domain["train"]["train-destination"] = multiwoz_utils.SlotInfo(
        is_categorical=True, possible_values=["leicester", "cambridge"]
    )

    examples = create_multiwoz21_trade_schemaless_data.create_schemaless_data(
        multiwoz_data.train_json,
        schema_info,
        multiwoz_data.slot_descriptions,
        create_multiwoz21_trade_schemaless_data.Options(
            description_type=common.DescriptionType("full_desc_with_domain"),
            delimiter=":",
            multiple_choice=common.MultipleChoiceFormat("none"),
            use_active_domains_only=False,
            blocked_domains=set(["hotel"]),
        ),
    )

    assert len(examples) == 4
    assert examples[1].src == (
        "0:restaurant-area or place of the restaurant 1:bus-destination of bus "
        "2:train-departure location of the train 3:restaurant-name of the "
        "restaurant 4:attraction-name of the attraction 5:bus-leaving time of "
        "bus 6:hotel-name of the hotel 7:restaurant-time of the restaurant "
        "booking 8:train-number of people booking for train 9:restaurant-food "
        "type for the restaurant 10:taxi-departure location of taxi 11:bus-day "
        "to use the bus tickets 12:train-day of the train 13:hotel-what is the "
        "type of the hotel 14:train-leaving time for the train 15:taxi-arrival "
        "time of taxi 16:hotel-length of stay at the hotel 17:train-arrival "
        "time of the train 18:taxi-leaving time of taxi 19:hotel-star rating "
        "of the hotel 20:hotel-number of people for the hotel booking "
        "21:hotel-price budget of the hotel 22:hotel-parking facility at the "
        "hotel 23:hotel-internet option at the hotel 24:restaurant-day of the "
        "restaurant booking 25:attraction-type of the attraction "
        "26:restaurant-price budget for the restaurant 27:hotel-area or place "
        "of the hotel 28:train-destination of the train 29:bus-departure "
        "location of bus 30:hospital-name of hospital department 31:hotel-day "
        "of the hotel booking 32:attraction-area or place of the attraction "
        "33:restaurant-number of people booking the restaurant "
        "34:taxi-destination of taxi [user] hi , can you help me find a train "
        "departing from cambridge ? [system] what is your destination ? [user] "
        "i am going to leicester"
    )
    assert examples[1].tgt == "[states] 2:cambridge 28:leicester [intents] [req_slots]"
    assert examples[1].dialog_id == "mul0708.json"
    assert examples[1].turn == 1


def test_blocked_many_domains(schema_file, tmp_path):
    multiwoz_data = multiwoz_utils.load_data(
        data_path=tmp_path, multiwoz_version="2.1", is_trade=True
    )
    schema_info = multiwoz_utils.load_schema(schema_file)

    # The testdata example doesn't have any categorical slots, so mock some.
    schema_info.slots_by_domain["train"]["train-departure"] = multiwoz_utils.SlotInfo(
        is_categorical=True, possible_values=["leicester", "cambridge"]
    )
    schema_info.slots_by_domain["train"]["train-destination"] = multiwoz_utils.SlotInfo(
        is_categorical=True, possible_values=["leicester", "cambridge"]
    )

    examples = create_multiwoz21_trade_schemaless_data.create_schemaless_data(
        multiwoz_data.train_json,
        schema_info,
        multiwoz_data.slot_descriptions,
        create_multiwoz21_trade_schemaless_data.Options(
            description_type=common.DescriptionType("full_desc_with_domain"),
            delimiter=":",
            multiple_choice=common.MultipleChoiceFormat("none"),
            use_active_domains_only=False,
            blocked_domains=set(["hotel", "train"]),
        ),
    )

    assert not examples


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"]))
