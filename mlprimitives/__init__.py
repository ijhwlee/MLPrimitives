# -*- coding: utf-8 -*-

"""Top-level package for MLPrimitives."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.3.5.dev0'

import os

from mlblocks import MLBlock

MLBLOCKS_PRIMITIVES = os.path.join(os.path.dirname(__file__), 'primitives')
MLBLOCKS_PIPELINES = os.path.join(os.path.dirname(__file__), 'pipelines')


def load_primitive(primitive, arguments=None):
    arguments = arguments or dict()

    print("[DEBUG-hwlee]mlprimitives.__init__.load_primitive: primitive = {0}".format(primitive))
    return MLBlock(primitive, **arguments)
