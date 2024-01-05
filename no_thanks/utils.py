# -*- coding: utf-8 -*-

"""Utility functions."""

from itertools import tee
from typing import Iterable, Tuple, TypeVar

EntityT = TypeVar("EntityT")


def pairwise(iterable: Iterable[EntityT]) -> Iterable[Tuple[EntityT, EntityT]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    it1, it2 = tee(iterable)
    next(it2, None)
    return zip(it1, it2)
