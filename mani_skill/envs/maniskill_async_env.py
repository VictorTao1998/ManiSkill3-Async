import gymnasium as gym
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from gymnasium.core import Env, ObsType
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
from gymnasium.vector.async_vector_env import AsyncVectorEnv, AsyncState
import torch
from numpy.typing import NDArray
import numpy as np
from typing import Iterable

class ManiSkillAsyncEnv(AsyncVectorEnv):
    def __init__(
            self,
            env_fns: Sequence[Callable[[], Env]],
            observation_space: Optional[gym.Space] = None,
            action_space: Optional[gym.Space] = None,
            shared_memory: bool = True,
            copy: bool = True,
            context: Optional[str] = None,
            daemon: bool = True,
            worker: Optional[Callable] = None,
    ):
        super(ManiSkillAsyncEnv, self).__init__(env_fns,
                                                observation_space,
                                                action_space,
                                                shared_memory,
                                                copy,
                                                context,
                                                daemon, worker)

    def call(self, name: str, split=False, *args, **kwargs) -> List[Any]:
        self.call_async(name, split, *args, **kwargs)
        return self.call_wait()

    def call_async(self, name: str, split=False, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            split: Whether to split the method or not.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        if split:
            for i, pipe in enumerate(self.parent_pipes):
                pipe.send(("_call", (name, tuple(a[i] for a in args), {k: v[i] for k, v in kwargs.items()})))
        else:
            for i, pipe in enumerate(self.parent_pipes):
                pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def reset(
            self,
            *,
            seed: Optional[Union[int, List[int]]] = None,
            options: Optional[dict] = None,
    ):
        self.reset_async(seed=seed, options=options)
        output = self.reset_wait(seed=seed, options=options)
        output = (shrink_dim(np2tensor(output[0]), self.num_envs),
                  shrink_dim(np2tensor(output[1]), self.num_envs))

        return output


    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        self._assert_is_running()

        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
                self._state.value,
            )

        for idx, pipe, single_seed in zip(range(self.num_envs), self.parent_pipes, seed):

            single_kwargs = {}
            if single_seed is not None:
                single_kwargs["seed"] = single_seed
            if options is not None:
                if "env_idx" in options:
                    if idx in options["env_idx"]:
                        single_kwargs["options"] = dict(env_idx=[0])
                    else:
                        single_kwargs["options"] = dict(env_idx=[])
                else:
                    single_kwargs["options"] = options

            pipe.send(("reset", single_kwargs))
        self._state = AsyncState.WAITING_RESET

    def step(
            self, actions
    ):
        self.step_async(actions)
        output = self.step_wait()
        # print("-----------steo_cnt-------------", output[4]['elapsed_steps'][0].item())
        is_none = [checkNone(o) for o in output]
        if torch.Tensor(is_none).any():
            assert 0
        output = tuple(shrink_dim(np2tensor(o), self.num_envs) for o in output)


        return output

def checkNone(in_dict):
    if isinstance(in_dict, dict):
        for k, v in in_dict.items():
            result = checkNone(v)
            if result:
                return result
        return False
    elif isinstance(in_dict, np.ndarray) and isinstance(in_dict.flat[0], dict):
        for d in in_dict:
            result = checkNone(d)
            if result:
                return result
        return False
    if isinstance(in_dict, torch.Tensor):
        for e in in_dict:
            if e is None:
                return True
        return False
    elif isinstance(in_dict, np.ndarray):
        for e in in_dict:
            if e is None:
                return True
        return False

def np2tensor(in_dict):
    if isinstance(in_dict, np.ndarray):
        if isinstance(in_dict.flat[0], torch.Tensor):
            if in_dict.flat[0].dtype == torch.int32:
                if None in in_dict:
                    pass
                in_dict = in_dict.astype(np.int32)
            elif in_dict.flat[0].dtype == torch.float32:
                in_dict = in_dict.astype(np.float32)
            elif in_dict.flat[0].dtype == torch.bool:
                in_dict = in_dict.astype(bool)
            else:
                raise NotImplementError("unsupported dtype")
        elif isinstance(in_dict.flat[0], dict):
            in_dict = combine_dict(list(in_dict))
            return np2tensor(in_dict)

        return torch.tensor(in_dict)
    for k, v in in_dict.items():
        if isinstance(v, dict):
            result = np2tensor(v)
        elif isinstance(v, np.ndarray):
            result = np2tensor(v)
        else:
            result = v
        in_dict[k] = result
    return in_dict

def combine_dict(dict_list):
    out_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, dict):
            result = combine_dict([d[k] for d in dict_list])
        elif isinstance(v, torch.Tensor):
            result = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            raise NotImplementError(f"unsupported dtype {type(v)}")
        out_dict[k] = result
    return out_dict


def shrink_dim(in_dict, num_envs):
    if isinstance(in_dict, torch.Tensor):
        out_tensor = in_dict.squeeze()
        if out_tensor.shape[0] != num_envs and num_envs == 1:
            out_tensor = out_tensor[None,...]
        if out_tensor.shape[0] != num_envs:
            raise NotImplementError("error return")
        if in_dict.shape[-1] == 1:
            out_tensor = out_tensor[..., None]
        return out_tensor
    for k, v in in_dict.items():
        if isinstance(v, (torch.Tensor, dict)):
            result = shrink_dim(v, num_envs)
        else:
            result = v
        in_dict[k] = result
    return in_dict
