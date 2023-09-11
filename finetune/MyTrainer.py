
import os, time
import json
import torch
import numpy as np
import torch.nn as nn
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal
from transformers.trainer import PredictionOutput
from transformers import Seq2SeqTrainer, TrainerCallback, Seq2SeqTrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.modeling_utils import unwrap_model, PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

import logging
from dataclasses import asdict, dataclass, field
from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)
from transformers.trainer import TRAINER_STATE_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
VALUE_HEAD_FILE_NAME = "value_head.bin"
FINETUNING_ARGS_NAME = "finetuning_args.json"


def load_pretrained() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        trust_remote_code=True,
        cache_dir = None,
        revision = "main",
        use_auth_token = None,
        user_fast = False,
        padding_side="right",
    )
    # Load and prepare pretrained models (without valuehead).
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        trust_remote_code=True,
        cache_dir = None,
        revision="main",
        use_auth_token=None,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="baichuan-inc/Baichuan-13B-Chat",
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir = None,
        revision="main",
        use_auth_token=None,
    )

    # # Register auto class to save the custom code files.
    if hasattr(config, "auto_map") and "AutoConfig" in config.auto_map:
        config.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
        tokenizer.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoModelForCausalLM" in config.auto_map:
        model.__class__.register_for_auto_class()

    # # Initialize adapters
    model = prepare_model_for_training(model)
    # for n, m in model.named_modules():
    #     with open("./layers0.txt","a") as f:
    #         f.write(str(n) +"\t" + str(type(m))+"\n")
    
    lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32.0,
            lora_dropout=0.1,
            target_modules=["W_pack"],
        )
    model = get_peft_model(model, lora_config)

    # for n, m in model.named_modules():
    #     with open("./layers1.txt","a") as f:
    #         f.write(str(n) +"\t" + str(type(m))+"\n")
    
    return model, tokenizer

def get_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]: # get state dict containing trainable parameters
    state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in model.named_parameters():
        if v.requires_grad:
            filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

    return filtered_state_dict


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
    if not os.path.exists(valuehead_file):
        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
    model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))
    return True


def prepare_model_for_training(
        model: PreTrainedModel,
        output_embedding_layer_name: Optional[str] = "lm_head",
        use_gradient_checkpointing: Optional[bool] = True,
        layer_norm_names: Optional[List[str]] = ["norm", "ln_f", "ln_attn", "ln_mlp"] # for LLaMA, BLOOM and Falcon settings
) -> PreTrainedModel:

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer: torch.nn.Linear = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model

def smooth(scalars: List[float], weight: Optional[float] = 0.9) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_loss(save_dictionary: os.PathLike, keys: Optional[List[str]] = ["loss"]) -> None:
    import matplotlib.pyplot as plt
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        plt.savefig(os.path.join(save_dictionary, "training_{}.png".format(key)), format="png", dpi=100)
        print("Figure saved:", os.path.join(save_dictionary, "training_{}.png".format(key)))


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    name_module_trainable: Optional[Literal["mlp", "self_attn", "self_attention"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for Freeze fine-tuning. \
                  LLaMA choices: [\"mlp\", \"self_attn\"], \
                  BLOOM & Falcon choices: [\"mlp\", \"self_attention\"], \
                  Baichuan choices: [\"mlp\", \"self_attn\"]"}
    )

    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )

    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning (similar with the learning rate)."}
    )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )

    lora_target: Optional[str] = field(
        default=""
    )

    def __post_init__(self):
        if isinstance(self.lora_target, str): # support custom target modules/layers of LoRA
            self.lora_target = [target.strip() for target in self.lora_target.split(",")]

    def save_to_json(self, json_path: str):
        """Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


class PeftTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
    """

    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if self.is_world_process_zero() and os.path.exists(os.path.join(self.args.output_dir, "trainer_log.jsonl")):
            logger.warning("Previous log file in this folder will be deleted.")
            os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoint.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = unwrap_model(self.model)

        if hasattr(model, "pretrained_model"): # for models with valuehead (currently using LoRA only)
            backbone_model = getattr(model, "pretrained_model")
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
        else:
            backbone_model = model

        backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))

        with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string() + "\n")
        self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))

    def _load_best_model(self):
        r"""
        Loads trainable parameters from model checkpoint.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")

        model = unwrap_model(self.model)
        backbone_model = getattr(model, "pretrained_model") if hasattr(model, "pretrained_model") else model

        backbone_model.load_adapter(self.state.best_model_checkpoint, getattr(backbone_model, "active_adapter"))
        if hasattr(model, "v_head") and load_valuehead_params(model, self.state.best_model_checkpoint):
            model.v_head.load_state_dict({
                "summary.weight": getattr(model, "reward_head_weight"),
                "summary.bias": getattr(model, "reward_head_bias")
            })


class Seq2SeqPeftTrainer(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
        inputs["labels"] = torch.cat((inputs["labels"], torch.zeros_like(inputs["input_ids"])[:, label_len:]), dim=-1)

        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        generated_tokens = generated_tokens[:, prompt_len:] if generated_tokens is not None else None

        return (loss, generated_tokens, labels)

    def save_predictions(
            self,
            predict_results: PredictionOutput
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))


if __name__=="__main__":

    from MyDataset import MyDataset
    from transformers import  DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM, AutoConfig

    model, tokenizer = load_pretrained()

    
    
    mydataset = MyDataset(
        dataset_path="../data", 
        dataset_name=os.listdir("../data"),
        tokenizer=tokenizer
    )
    dataset = mydataset.build_dataset()
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX,
    )
    finetuning_args = FinetuningArguments(
        lora_target="W_pack",
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir="../output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_steps=1000,
        learning_rate=5e-5,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
    )
    trainer = Seq2SeqPeftTrainer(
        finetuning_args=finetuning_args,
        model=model,
        train_dataset=dataset,
        args=training_args,
        # eval_dataset = 
        # args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[LogCallback()],
        # compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()
    if trainer.is_world_process_zero():
        plot_loss("../output", keys=["loss", "eval_loss"])