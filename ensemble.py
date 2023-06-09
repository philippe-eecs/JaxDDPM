from typing import Type
import flax.linen as nn


class ParallelEnsemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(self.net_cls,
                           variable_axes={'params': 0},
                           split_rngs={
                               'params': True,
                               'dropout': True
                           },
                           in_axes=0,
                           out_axes=0,
                           axis_size=self.num)
        return ensemble()(*args, **kwargs)