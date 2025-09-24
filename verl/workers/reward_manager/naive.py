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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict

from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from collections import defaultdict
import ray

import time
from am_reward_acc import compute_score

@ray.remote
def compute_score_remote(i, response_str, ground_truth, data_source, extra_info, valid_response_length, prompt_str,is_complete):
    
    score = compute_score(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        is_complete=is_complete,
    )
    return i, score, valid_response_length, data_source, prompt_str, response_str, ground_truth


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        # start = time.time()

        # 准备输入数据：确保是可序列化的 dict
        serialized_inputs = []
        max_resp_len = 0
        for i in range(len(data)):
            item = data[i]

            prompt_ids = item.batch['prompts']
            attn_mask = item.batch['attention_mask']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = attn_mask[:prompt_length].sum().item()
            valid_prompt_ids = prompt_ids[-int(valid_prompt_length):]

            response_ids = item.batch['responses']
            valid_response_length = attn_mask[prompt_length:].sum().item()
            valid_response_ids = response_ids[:int(valid_response_length)]
            max_resp_len = max(len(valid_response_ids),max_resp_len)
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
            data_source = item.non_tensor_batch[self.reward_fn_key]
            extra_info = item.non_tensor_batch.get('extra_info', None)

            is_complete = item.non_tensor_batch.get('is_complete', None)
            serialized_inputs.append({
                "i": i,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
                "valid_response_length": valid_response_length,
                "prompt_str": prompt_str,
                "is_complete":is_complete
            })
        futures = [
            compute_score_remote.remote(
                item["i"],
                item["response_str"],
                item["ground_truth"],
                item["data_source"],
                item["extra_info"],
                item["valid_response_length"],
                item["prompt_str"],
                item['is_complete'],
            )
            for item in serialized_inputs
        ]

        results = ray.get(futures)  # 等待所有任务完成
        print("max_resp_len:",max_resp_len)
        for i, score, valid_response_length, data_source, prompt_str, response_str, ground_truth in results:
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, int(valid_response_length) - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
        # end = time.time()
        # print("耗时：", end - start, "秒")
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor



# 原始版本
# class NaiveRewardManager:
#     """The reward manager.
#     """

#     def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
#         self.tokenizer = tokenizer
#         self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#         self.compute_score = compute_score or _default_compute_score
#         self.reward_fn_key = reward_fn_key

#     def __call__(self, data: DataProto, return_dict=False):
#         """We will expand this function gradually based on the available datasets"""

#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if 'rm_scores' in data.batch.keys():
#             if return_dict:
#                 return {"reward_tensor": data.batch['rm_scores']}
#             else:
#                 return data.batch['rm_scores']

#         reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
#         reward_extra_info = defaultdict(list)

#         already_print_data_sources = {}
#         # breakpoint()
#         for i in range(len(data)):

#             data_item = data[i]  # DataProtoItem

#             prompt_ids = data_item.batch['prompts']

#             prompt_length = prompt_ids.shape[-1]

#             valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
#             valid_prompt_ids = prompt_ids[-valid_prompt_length:]

#             response_ids = data_item.batch['responses']
#             valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
#             valid_response_ids = response_ids[:valid_response_length]

#             # decode
#             prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
#             response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

#             ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

#             data_source = data_item.non_tensor_batch[self.reward_fn_key]

#             extra_info = data_item.non_tensor_batch.get('extra_info', None)

#             score = self.compute_score(
#                 data_source=data_source,
#                 solution_str=response_str,
#                 ground_truth=ground_truth,
#                 extra_info=extra_info,
#             )

#             if isinstance(score, dict):
#                 reward = score["score"]
#                 # Store the information including original reward
#                 for key, value in score.items():
#                     reward_extra_info[key].append(value)
#             else:
#                 reward = score

#             reward_tensor[i, valid_response_length - 1] = reward

#             if data_source not in already_print_data_sources:
#                 already_print_data_sources[data_source] = 0

#             if already_print_data_sources[data_source] < self.num_examine:
#                 already_print_data_sources[data_source] += 1
#                 print("[prompt]", prompt_str)
#                 print("[response]", response_str)
#                 print("[ground_truth]", ground_truth)
#                 if isinstance(score, dict):
#                     for key, value in score.items():
#                         print(f"[{key}]", value)
#                 else:
#                     print(f"[score]", score)

#         if return_dict:
#             return {
#                 "reward_tensor": reward_tensor,
#                 "reward_extra_info": reward_extra_info,
#             }
#         else:
#             return reward_tensor
