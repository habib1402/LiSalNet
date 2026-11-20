#!/usr/bin/env python3
"""Dependency-free tests of strict checkpoint-key/shape migration logic.

Uses shape-only stand-ins; does NOT test PyTorch deserialization or tensor values.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_utils import REMOVED_SHAPES, prepare_state_dict


class ShapeOnly:
    def __init__(self, *shape):
        self.shape = shape


def main():
    expected = {'Backbone.features.0.weight': ShapeOnly(32, 3, 3, 3), 'cls1.bias': ShapeOnly(1)}
    legacy = {**expected, **{key: ShapeOnly(*shape) for key, shape in REMOVED_SHAPES.items()}}
    count = 0
    for candidate, number_removed in [
        (expected, 0), (legacy, 6), ({'state_dict': legacy, 'epoch': 7}, 6),
        ({'model_state_dict': {'module.' + key: value for key, value in legacy.items()}}, 6),
    ]:
        result, removed = prepare_state_dict(candidate, expected)
        assert set(result) == set(expected) and len(removed) == number_removed
        assert all(result[key] is expected[key] for key in expected)
        count += 1
    invalid = [
        {'cls1.bias': expected['cls1.bias']},
        {**legacy, 'unknown.weight': ShapeOnly(1)},
        {**legacy, 'cls1.bias': ShapeOnly(2)},
        {**legacy, 'Backbone.features.18.0.weight': ShapeOnly(1)},
        {**legacy, 'Backbone.features.18.extra': ShapeOnly(1)},
        {**legacy, 'module.cls1.bias': ShapeOnly(1)},
        {'epoch': 5}, {},
    ]
    for candidate in invalid:
        try:
            prepare_state_dict(candidate, expected)
        except (TypeError, ValueError):
            count += 1
        else:
            raise AssertionError('Malformed checkpoint was accepted.')
    print(f'Passed {count} checkpoint key/shape migration cases (no PyTorch execution).')


if __name__ == '__main__':
    main()
