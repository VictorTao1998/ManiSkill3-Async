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


def split_dict(concat_dict, num, keep_dim=True):
    out_dict_list = [{} for _ in range(num)]
    for k, v in concat_dict.items():
        if isinstance(v, dict):
            result = split_dict(v, num)
        elif isinstance(v, torch.Tensor):
            assert v.shape[0] == num
            if keep_dim:
                result = [v[i:i + 1] for i in range(num)]
            else:
                result = [v[i] for i in range(num)]
        else:
            raise NotImplementedError
        for i in range(num):
            out_dict_list[i][k] = result[i]
    return out_dict_list
