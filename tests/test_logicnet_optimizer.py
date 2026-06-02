import yaml
from src.train_diffusion import DiffusionTrainingConfig, DiffusionTrainer


def test_logicnet_optimizer():
    with open("configs/zelda_hmolqd.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    config = DiffusionTrainingConfig.from_dict(config_dict)
    config["stage"] = "logicnet"
    trainer = DiffusionTrainer(config)

    assert hasattr(trainer.model.guidance, "logic_net")
    assert trainer.model.guidance.logic_net is trainer.logic_net
    assert "logic_net" not in getattr(trainer.model.guidance, "_modules", {})
    assert not any("guidance.logic_net" in key for key in trainer.diffusion.state_dict())

    logic_net = trainer.logic_net
    logic_net_params = list(logic_net.parameters())
    optimizer_param_ids = {
        id(param)
        for group in trainer.optimizer.param_groups
        for param in group["params"]
    }
    group_names = {group.get("name") for group in trainer.optimizer.param_groups}

    assert logic_net_params
    assert {"diffusion", "condition_encoder", "logic_net"} <= group_names
    assert all(param.requires_grad for param in logic_net_params)
    assert {id(param) for param in logic_net_params} <= optimizer_param_ids
    assert trainer._estimated_total_steps > 1

    logic_group = next(group for group in trainer.optimizer.param_groups if group.get("name") == "logic_net")
    assert logic_group["lr"] < logic_group["base_lr"]

if __name__ == "__main__":
    test_logicnet_optimizer()
