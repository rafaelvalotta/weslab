# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import numpy as np
from collections import namedtuple
from wesl.optimizer.interarray.fileio import file2graph,file2graph_topfarm
from wesl.optimizer.interarray.synthetic import synthfarm2graph, equidistant
from pathlib import Path


def namedtuplify(namedtuple_typename='', **kwargs):
    NamedTuplified = namedtuple(namedtuple_typename,
                                tuple(str(kw) for kw in kwargs))
    return NamedTuplified(**kwargs)

def namedtuplify_topfarm(namedtuple_typename='', **kwargs):
    NamedTuplified = namedtuple(namedtuple_typename,
                                tuple(str(kw) for kw in kwargs))
    return NamedTuplified(**kwargs)

def tess(radius, spacing=1000):
    NodeC = equidistant(radius, center='centroid', spacing=spacing)
    RootC = np.zeros((1, 2), dtype=float)
    return synthfarm2graph(RootC, NodeC, name='SynthTess')


def tess3(radius, spacing=1000):
    h = np.sqrt(3)/2
    NodeC = equidistant(radius, center='vertex', spacing=spacing)
    RootC = 5*spacing*np.array(((3/4, -h/2), (-3/4, -h/2), (0., h)))
    return synthfarm2graph(RootC, NodeC, name='SynthTess (3 OSS)')


def tess3sm(radius, spacing=1000):
    h = np.sqrt(3)/2
    NodeC = equidistant(radius, center='vertex', spacing=spacing)
    RootC = 2.5*spacing*np.array(((-0.5, -h), (-0.5, h), (1., 0.)))
    return synthfarm2graph(RootC, NodeC, name='SynthTess small (3 OSS)')

datapath = Path(__file__).resolve().parent.parent / 'data'

g = namedtuplify(
    namedtuple_typename='FarmGraphs',
    # .xlsx files
    # 100 WTG
    thanet=file2graph(datapath / 'Thanet.xlsx'),
    # 80 WTG
    dantysk=file2graph(datapath / 'DanTysk.xlsx'),
    # 80 WTG
    horns=file2graph(datapath / 'Horns Rev 1.xlsx'),
    # 111 WTG
    anholt=file2graph(datapath / 'Anholt.xlsx'),
    # 108 WTG
    sands=file2graph(datapath / 'West of Duddon Sands.xlsx'),
    # 30 WTG
    ormonde=file2graph(datapath / 'Ormonde.xlsx'),
    # 175 WTG, 2 OSS
    london=file2graph(datapath / 'London Array.xlsx'),
    # 27 WTG
    rbn=file2graph(datapath / 'BIG Ronne Bank North.xlsx'),
    # 53 WTG
    rbs=file2graph(datapath / 'BIG Ronne Bank South.xlsx'),

    # synthetic farms
    # 114 WTG
    tess=tess(radius=5600),
    # 241 WTG, 3 OSS
    tess3=tess3(radius=8000),
    # ? WTG, 3 OSS
    # tess3sm=tess3sm(radius=5300),
)

# g1 = namedtuplify_topfarm(
#     namedtuple_typename='FarmGraphs',
#     # .xlsx files
#     # 100 WTG
#     # 80 WTG
#     horns=file2graph_topfarm(datapath / 'Horns Rev 1.xlsx',x,y),

#     # synthetic farms
#     # 114 WTG
#     tess=tess(radius=5600),
#     # 241 WTG, 3 OSS
#     tess3=tess3(radius=8000),
#     # ? WTG, 3 OSS
#     # tess3sm=tess3sm(radius=5300),
# )


def g1(x,y,boundary):
    
    """
    
    g (): 
    
    """

    g = namedtuplify_topfarm(
    namedtuple_typename='FarmGraphs',
    # .xlsx files
    # 100 WTG
    # 80 WTG
    horns=file2graph_topfarm(datapath / 'Horns Rev 1.xlsx',x,y,boundary),

    # synthetic farms
    # 114 WTG
    tess=tess(radius=5600),
    # 241 WTG, 3 OSS
    tess3=tess3(radius=8000),
    # ? WTG, 3 OSS
    # tess3sm=tess3sm(radius=5300),
)
    return g
