
import torch
from . import _C


#
def position_to_mortoncode(position, octree_level):
    assert torch.is_tensor(position) and torch.is_tensor(octree_level)
    assert len(position.shape) == 2 and position.shape[1] == 3
    assert position.dtype == torch.int64
    assert octree_level.dtype == torch.int8

    return _C.position_to_mortoncode(position, octree_level)


#
def mortoncode_to_position(mortoncode, octree_level):
    assert torch.is_tensor(mortoncode) and torch.is_tensor(octree_level)
    assert mortoncode.dtype == torch.int64
    assert octree_level.dtype == torch.int8

    return _C.mortoncode_to_position(mortoncode, octree_level)
