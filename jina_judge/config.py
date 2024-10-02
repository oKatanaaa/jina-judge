from dataclasses import dataclass, field, _MISSING_TYPE
from typing import List, Union, Optional, Any
import yaml
import os
from comet_ml import Experiment
from copy import deepcopy


@dataclass
class TrainConfig:
    output_dir: str = "experiments"
    checkpoint: Union[str, None] = None
    train_dataset: Union[str, None] = None
    test_dataset: Union[str, None] = None
    val_dataset: Union[str, None] = None
    epochs: int = 10
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: Union[float, None] = 1.0
    learning_rate: float = 1e-5
    dropout: float = 0.1
    weight_decay: float = 1e-4
    all_params: bool = True
    warmup_steps: int = 0
    max_ctx_len: int = 4096
    device: str = "cuda:0"
    comet_api_key: Union[str, None] = None
    comet_project_name: str = "jina-judge"
    comet_workspace: Union[str, None] = None
    # Will be initialized during loading
    experiment: Experiment = None


def load_config(config_path) -> tuple[dict, TrainConfig]:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    _config = dict([(k, v) for k, v in config_dict.items() if v is not None])
    # Check data integrity
    fields = TrainConfig.__dataclass_fields__
    missing_keys = list(set(fields.keys()) - set(_config.keys()))
    if len(missing_keys) > 0:
        for k in missing_keys:
            default_val = fields[k].default
            if isinstance(default_val, _MISSING_TYPE):
                default_val = fields[k].default_factory()
            print(f'WARNING! Missing key: {k}. Setting to default value: {default_val}')
            _config[k] = default_val

    # ------------- Setup comet ml -----------------

    if config_dict.get("experiment") is not None:
        raise ValueError("experiment should be None, as it will be initialized during loading")
    
    if _config["comet_api_key"] is not None:
        assert _config["comet_project_name"] is not None, "comet_project_name must be set"
        assert _config["comet_workspace"] is not None, "comet_workspace must be set"

        _config["experiment"] = Experiment(
            api_key=_config["comet_api_key"],
            project_name=_config["comet_project_name"],
            workspace=_config["comet_workspace"])
    
    if os.environ.get("COMET_API_KEY", None) is not None:
        assert _config["comet_project_name"] is not None, "comet_project_name must be set"
        assert _config["comet_workspace"] is not None, "comet_workspace must be set"

        _config["experiment"] = Experiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name=_config["comet_project_name"],
            workspace=_config["comet_workspace"])
    
    if _config["experiment"] is not None:
        # Log hyperparameters
        _config_copy = deepcopy(config_dict)
        _config_copy["comet_api_key"] = None
        _config_copy["comet_project_name"] = None
        _config_copy["comet_workspace"] = None
        _config["experiment"].log_parameters(_config_copy)

    # ------------- Setup comet ml -----------------

    training_config = TrainConfig(**_config)
            
    return config_dict, training_config


def save_config(config_dict: dict):
    output_dir = config_dict['output_dir']
    with open(os.path.join(output_dir, 'jina-judge.yaml'), 'w') as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
        print('Saved config in the output directory.')
