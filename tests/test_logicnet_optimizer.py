import yaml
import torch
from src.train_diffusion import DiffusionTrainingConfig, DiffusionTrainer
from src.core.logic_net import LogicNet

def test_logicnet_optimizer():
    # 1. Load config
    with open("configs/zelda_hmolqd.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    
    # 2. Convert to config object
    config = DiffusionTrainingConfig.from_dict(config_dict)
    
    # Force stage to logicnet just in case
    config["stage"] = "logicnet"
    
    # 2. Create Trainer
    print("Initializing DiffusionTrainer...")
    trainer = DiffusionTrainer(config)
    
    model = trainer.model
    optimizer = trainer.optimizer
    
    # 3. Check LogicNet
    if not hasattr(model.guidance, "logic_net") or model.guidance.logic_net is None:
        print("ERROR: logic_net not found in model.guidance")
        return

    logic_net = model.guidance.logic_net
    logic_net_params = list(logic_net.parameters())
    num_logic_params = len(logic_net_params)
    total_logic_elements = sum(p.numel() for p in logic_net_params)
    
    print(f"LogicNet parameter blocks: {num_logic_params}")
    print(f"LogicNet total elements: {total_logic_elements}")
    
    # Check requires_grad
    requires_grad_count = sum(1 for p in logic_net_params if p.requires_grad)
    print(f"LogicNet params with requires_grad=True: {requires_grad_count} / {num_logic_params}")
    
    # Check optimizer
    opt_param_ids = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            opt_param_ids.add(id(p))
    
    logic_param_ids = set(id(p) for p in logic_net_params)
    overlap = logic_param_ids.intersection(opt_param_ids)
    
    print(f"LogicNet params found in Optimizer: {len(overlap)} / {num_logic_params}")
    
    if len(overlap) == 0:
        print("FAILURE: LogicNet parameters are MISSING from the optimizer.")
    elif len(overlap) < num_logic_params:
        print("PARTIAL FAILURE: Only some LogicNet parameters are in the optimizer.")
    else:
        print("SUCCESS: All LogicNet parameters are in the optimizer.")

if __name__ == "__main__":
    test_logicnet_optimizer()
