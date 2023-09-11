

from typing import List, Optional
import os
from transformers import Seq2SeqTrainingArguments, AutoTokenizer

from datasets import Dataset, load_dataset, concatenate_datasets
from dataclasses import dataclass
from transformers.tokenization_utils import PreTrainedTokenizer


IGNORE_INDEX = -100
@dataclass
class Template:
    name: str
    def __post_init__(self):
        if self.name == "default":
            r"""
            Supports: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
            """
            self._register_template(
                prefix="",
                prompt="<reserved_102>{query}<reserved_103>",
                sep="",
                use_history=True
            )
        else:
            raise ValueError("Template {} does not exist.".format(self.name))

    def get_prompt(self, query: str, history: Optional[list] = None, prefix: Optional[str] = "") -> str:
        r"""
        Returns a string containing prompt without response.
        """
        return "".join(self._format_example(query, history, prefix))

    def get_dialog(self, query: str, resp: str, history: Optional[list] = None, prefix: Optional[str] = "") -> List[str]:
        r"""
        Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.
        """
        return self._format_example(query, history, prefix) + [resp]

    def _register_template(self, prefix: str, prompt: str, sep: str, use_history: Optional[bool] = True) -> None:
        self.prefix = prefix
        self.prompt = prompt
        self.sep = sep
        self.use_history = use_history

    def _format_example(self, query: str, history: Optional[list] = None, prefix: Optional[str] = "") -> List[str]:
        prefix = prefix if prefix else self.prefix          # use prefix if provided
        prefix = prefix + self.sep if prefix else ""        # add separator for non-empty prefix
        
        history = history if (history and self.use_history) else []
        history = history + [(query, "<dummy>")]
        convs = []
        for turn_idx, (user_query, bot_resp) in enumerate(history):
            if turn_idx == 0:
                convs.append(prefix + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs[:-1] # drop last

@dataclass
class DatasetAttr:
    dataset_name: Optional[str] = None

    def __repr__(self) -> str:
        return self.dataset_name

    def __post_init__(self):
        self.prompt_column = "instruction"
        self.query_column = "input"
        self.response_column = "output"
        self.history_column = None


class MyDataset:
    def __init__(self, dataset_path, dataset_names, tokenizer) -> None:
        self.dataset_names= dataset_names
        self.dataset_path = dataset_path
        self.prompt_template = Template("default")
        self.max_source_length = 256 
        self.max_target_length = 512
        self.tokenizer = tokenizer
        self.predict_with_generate = False
    def get_dialog(self, examples):
        for i in range(len(examples["prompt"])): # 1000
            if examples["prompt"][i] and examples["response"][i]:
                query, answer = examples["prompt"][i], examples["response"][i]
                query = query + "\n" + examples["query"][i] if examples["query"][i] else query
                dialog = self.prompt_template.get_dialog(query=query, resp=answer, history=examples["history"][i], prefix="")
                yield dialog
    
    def preprocess_supervised_dataset(self, examples):
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for input with history, we build multiple input-label pairs just like:
        # https://github.com/lm-sys/FastChat/blob/f17c092f64840fa6354ed52789dccb2daa793d0b/fastchat/train/train.py#L112
        model_inputs = {"input_ids": [], "labels": []}
        max_length = self.max_source_length + self.max_target_length 

        for dialog in self.get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = self.tokenizer.encode(text=dialog[2*i], add_special_tokens=(i == 0))
                target_ids = self.tokenizer.encode(text=dialog[2*i+1], add_special_tokens=False)

                if len(source_ids) > self.max_source_length:
                    source_ids = source_ids[:self.max_source_length]
                if len(target_ids) > self.max_target_length - 1: # eos token
                    target_ids = target_ids[:self.max_target_length - 1]

                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [self.tokenizer.eos_token_id]
                labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [self.tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

        return model_inputs
    
    def preprocess_unsupervised_dataset(self, examples):
        # build inputs with format `<bos> X` and labels with format `<bos> Y`
        model_inputs = {"input_ids": [], "labels": []}

        for dialog in self.get_dialog(examples):
            prompt, answer = "".join(dialog[:-1]), dialog[-1]

            source_ids = self.tokenizer.encode(text=prompt, add_special_tokens=True)
            target_ids = self.tokenizer.encode(text=answer, add_special_tokens=True)

            if len(source_ids) > 256:
                source_ids = source_ids[:256]
            if len(target_ids) > 512:
                target_ids = target_ids[:512]

            model_inputs["input_ids"].append(source_ids)
            model_inputs["labels"].append(target_ids)

        return model_inputs
    
    def build_dataset(
        self,
    ) -> Dataset:
        all_datasets: List[Dataset] = [] # support multiple datasets

        for dataset_name in self.dataset_names:
            dataset_attr = DatasetAttr(dataset_name=dataset_name)
            data_files.append(os.path.join(self.dataset_path, dataset_attr.dataset_name))

            data_files = os.path.join(self.dataset_path, dataset_attr.dataset_name)


            extension = data_files.split(".")[-1]
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=None,
                use_auth_token= None
            )
            dataset = raw_datasets["train"]
            '''dataset
                - instruction:
                - input:
                - output:
            '''
            dummy_data = [None] * len(dataset)
            for column_name, target_name in [
                ("prompt_column", "prompt"),
                ("query_column", "query"),
                ("response_column", "response"),
                ("history_column", "history")
            ]: # every dataset will have 4 columns same as each other
                if getattr(dataset_attr, column_name) != target_name:
                    if getattr(dataset_attr, column_name):
                        dataset = dataset.rename_column(getattr(dataset_attr, column_name), target_name)
                    else: # None or empty string
                        dataset = dataset.add_column(target_name, dummy_data)

            '''dataset
                - prompt:   instruction
                - query:    input
                - response: output
                - history:  None
            '''
            column_names = list(dataset.column_names)

            preprocess_function = self.preprocess_unsupervised_dataset \
                if self.predict_with_generate else self.preprocess_supervised_dataset

            dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=5,
                remove_columns=column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset"
            )
            all_datasets.append(dataset)
            # return dataset
        if len(self.dataset_names) == 1:
            all_datasets=all_datasets[0]
        else:
            all_datasets = concatenate_datasets(all_datasets)
        return all_datasets



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        trust_remote_code=True,
        cache_dir = None,
        revision = "main",
        use_auth_token = None,
        user_fast = False,
        padding_side="right",
    )
    mydataset = MyDataset(
        dataset_path="/data/agl/Baichuan-13B-Finetuning/data", 
        dataset_name="alpaca_data_zh_51k.json",
        tokenizer=tokenizer)

    training_args = Seq2SeqTrainingArguments(output_dir="./output")

    dataset = mydataset.build_dataset(training_args=training_args)
