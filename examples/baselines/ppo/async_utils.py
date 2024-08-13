import torch


def concat_dict(dict_list):
    out_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, dict):
            result = concat_dict([d[k] for d in dict_list])
        elif isinstance(v, torch.Tensor):
            if v.shape[0] == 1:
                tensor_list = [d[k] for d in dict_list]
            else:
                tensor_list = [d[k][None, ...] for d in dict_list]
            result = torch.cat(tensor_list, dim=0)
        else:
            raise NotImplementedError
        out_dict[k] = result

    return out_dict
