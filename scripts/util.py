import sys
from pathlib import Path

PARENT_DIR = Path(sys.argv[0]).parent
PROJECT_DIR = Path('..' if PARENT_DIR == Path('.') else PARENT_DIR / '..').resolve()

import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.typing import ModuleID, TensorType
from ray.rllib.utils.annotations import override
from typing import Dict, Any, Optional

# Custom learner adding L1 weight regularization
class LassoLearner(PPOTorchLearner):

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:

        base_total_loss = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        # Compute the mean of all the RLModule's weights' absolute values.
        parameters = self.get_parameters(self.module[module_id])
        mean_abs_weight = torch.mean(torch.cat([p.reshape(-1).abs() for p in parameters]))

        self.metrics.log_value(
            key=(module_id, "lasso_coeff"),
            value=mean_abs_weight,
            window=1,
        )

        total_loss = (
            base_total_loss
            + config.learner_config_dict["lasso_coeff"] * mean_abs_weight
        )

        return total_loss