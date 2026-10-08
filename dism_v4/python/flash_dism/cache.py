"""FLA Cache with a local bridge for the installed Transformers layer API."""
import torch
from fla.models.utils import Cache, FLALayer


class _CompatibleLayer(FLALayer):
    def get_max_length(self):
        return self.get_max_cache_shape()

    def reorder_cache(self, beam_idx):
        if self.state is None:
            return
        for key, value in self.state.items():
            if isinstance(value, torch.Tensor):
                self.state[key] = value.index_select(0, beam_idx.to(value.device))
            elif isinstance(value, (tuple, list)):
                self.state[key] = tuple(
                    tensor.index_select(0, beam_idx.to(tensor.device))
                    if isinstance(tensor, torch.Tensor) else tensor for tensor in value)


class DismCache(Cache):
    """FLA Cache subclass; may be shared by DISM and GDN layers.

    No global monkey-patching. The adapter only supplies the missing legacy
    get_max_length method and batch reordering for FLA's dictionary states.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self, 'use_layer_class_to_replicate', False):
            self.layer_class_to_replicate = _CompatibleLayer
