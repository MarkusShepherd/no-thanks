# -*- coding: utf-8 -*-

"""Utility functions."""

from itertools import tee
from typing import Iterable, Tuple, TypeVar

TYPE = TypeVar("TYPE")


def pairwise(iterable: Iterable[TYPE]) -> Iterable[Tuple[TYPE, TYPE]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    it1, it2 = tee(iterable)
    next(it2, None)
    return zip(it1, it2)
