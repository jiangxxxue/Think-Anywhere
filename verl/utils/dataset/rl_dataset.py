# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import ListConfig
import os
from typing import List, Union
import copy
import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


SYSTEM_PROMPT = """
You are a coding assistant that generates both Python code and inline self-guidance signals.

First output <think>...</think> with brief reasoning, then output the final Python code.

MUST FOLLOW Rules for <thinkanywhere>...</thinkanywhere> tags (very important):
1. You MUST use <thinkanywhere>...</thinkanywhere> tags for self-guidance, intermediate reasoning, or local planning.
2. <thinkanywhere>...</thinkanywhere> MUST NOT appear on a standalone line. It must be embedded within an existing Python statement token sequence.
3. You MUST insert <thinkanywhere>...</thinkanywhere> tags frequently around important operations, but you do not need to annotate every line.
4. The code must remain valid and executable after removing all <thinkanywhere>...</thinkanywhere> segments.
5. Do NOT wrap a full-line comment in <thinkanywhere>...</thinkanywhere>. For example, avoid `<thinkanywhere># comment</thinkanywhere>`; instead, attach <thinkanywhere>...</thinkanywhere> to a real code statement.
""".strip()

def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_local_path_from_hdfs
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)
    
    def _get_full_chat_with_system(self, raw_prompt_chat):
        """封装拼接 System Prompt 的逻辑"""
        if isinstance(raw_prompt_chat, np.ndarray):
            raw_prompt_chat = raw_prompt_chat.tolist()
        chat = copy.deepcopy(raw_prompt_chat)
        chat.insert(0, {
            "content": SYSTEM_PROMPT,
            "role": "system"
        })
        return chat

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            if parquet_file.endswith(".pkl"):
                dataframe = pd.read_pickle(parquet_file)
                if not isinstance(dataframe, pd.core.frame.DataFrame):
                    dataframe = pd.DataFrame(dataframe)
            else:
                dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        def _calc_full_prompt_length(doc):
            # 1. 拼接 System Prompt
            full_chat = self._get_full_chat_with_system(doc[prompt_key])
            # 2. 计算拼接后的 token 长度
            return len(tokenizer.apply_chat_template(
                full_chat, 
                add_generation_prompt=True
            ))

        # 过滤逻辑：拼接 System Prompt 后长度 ≤ max_prompt_length
        self.dataframe = self.dataframe[
            self.dataframe.apply(_calc_full_prompt_length, axis=1) <= self.max_prompt_length
        ]

        print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        chat = copy.deepcopy(chat)
        if isinstance(chat, np.ndarray):
            chat = chat.tolist()
        chat.insert(0, {
            "content": SYSTEM_PROMPT,
            "role": "system"
        })

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
