import torch
import torch.nn.functional as F
import numpy as np
def discretized_actions(action, discretized_low, discretized_high):
    scaled_tensor = action * (discretized_high - discretized_low) + discretized_low

    discretized_tensor = torch.floor(scaled_tensor)
    diff = torch.tensor([discretized_high - discretized_low - torch.sum(row) for row in discretized_tensor])

    for row_idx, row_diff in enumerate(diff):
        if row_diff > 0:
            indices_to_increment = torch.topk(scaled_tensor[row_idx] - discretized_tensor[row_idx], int(row_diff))[1]
            discretized_tensor[row_idx, indices_to_increment] += 1
        elif row_diff < 0:
            indices_to_decrement = torch.topk(discretized_tensor[row_idx] - (scaled_tensor[row_idx] - 1), int(abs(row_diff)))[1]
            discretized_tensor[row_idx, indices_to_decrement] -= 1

    return discretized_tensor.long()

def get_optim_param(optimizer: torch.optim) -> list:
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list

def get_action_wrapper(func,
                       method = "softmax", # softmax, reweight
                       T = 1.0,
                       ):
    def get_action(x,
                   mask = None,mkt=None,
                   mask_value = 1e6,lambda_r = 0.3,
                   **kwargs):

        if method == "softmax":
            pred = func(x, **kwargs)
            weight = pred / T
            if mask is not None:
                mask_bool = mask_bools(mask, x, pred)
                mask_bool = mask_bool * mask_value
                weight = weight - mask_bool
                weight = (weight.squeeze(-1) + lambda_r*(1-mask_bool)* mkt).clamp(-1,1) # range [-1, 1]
            else:
                if mkt is not None:
                    weight = (weight.squeeze(-1) + lambda_r* mkt).clamp(-1,1) # range [-1, 1]

        else:
            raise NotImplementedError
        return weight
    return get_action

def get_action_logprob_wrapper(func,
                       method = "softmax", # softmax
                       T = 1.0,
                       ):
    def get_action(x,mkt=None,
                   mask = None,
                   mask_value = 1e6,
                   **kwargs):

        if method == "softmax":
            pred, logprob = func(x, **kwargs)
            weight = pred / T

            if mask is not None:
                mask_bool = mask_bools(mask, x, pred)
                mask_bool = mask_bool * mask_value
                pred = pred - mask_bool

        elif method == "reweight":
            pred, logprob = func(x, **kwargs)

            if mask is not None:
                mask_bool = mask_bools(mask, x, pred)
                mask_bool = mask_bool * mask_value
                pred = pred - mask_bool

            indices = torch.sort(pred)[1]
            soft_pred = pred * torch.log(indices + 1)

            weight = F.softmax(soft_pred, dim=-1).squeeze(-1)

        else:
            raise NotImplementedError
        return weight, logprob
    return get_action

def forward_action_wrapper(func,
                    method = "softmax", # softmax
                    T = 1.0,lambda_r = 0.3,
                    ):
    def forward_action(x,mkt=None,
                       mask = None,
                       mask_value = 1e6,
                       **kwargs):

        if method == "softmax":
            pred = func(x, **kwargs)
            weight = pred / T
            if mask is not None:
                mask_bool = mask_bools(mask, x, pred)
                mask_bool = mask_bool * mask_value
                weight = weight - mask_bool
                weight = (weight.squeeze(-1) + lambda_r*(1-mask_bool)* mkt).clamp(-1,1) # range [-1, 1]
            else:
                if mkt is not None:
                    weight = (weight.squeeze(-1) + lambda_r* mkt).clamp(-1,1) # range [-1, 1]
        else:
            raise NotImplementedError
        return weight

    return forward_action

def mask_bools(mask,x,pred):
    if pred.shape[1] == mask.shape[1]+1:
        # mask_tensor = torch.from_numpy(mask).to(x.device)
        mask_bool = torch.concat(
             [torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), mask], dim=1).float()
    else:
        mask_bool = mask.bool()
    return mask_bool
