# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull
from wesl.optimizer.interarray.interarraylib import make_graph_metrics, NodeTagger


F = NodeTagger()


def synthfarm2graph(RootC, NodeC, BoundaryC=None, name=''):
    N = NodeC.shape[0]
    M = RootC.shape[0]

    # build data structures
    VertexC = np.vstack((NodeC, RootC))
    if BoundaryC is None:
        hull = ConvexHull(VertexC)
        boundary = np.reshape(VertexC[hull.simplices],
                              (hull.simplices.shape[0]*2, 2))
        # this arctan2() trick only works because the origin is in the middle
        BoundaryC = boundary[np.argsort(np.arctan2(*boundary.T[::-1]))]

    # create networkx graph
    G = nx.Graph(M=M,
                 VertexC=VertexC,
                 boundary=BoundaryC,
                 name=name)
    G.add_nodes_from(((n, {'label': F[n], 'type': 'wtg'})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'type': 'oss'})
                      for r in range(-M, 0)))

    make_graph_metrics(G)
    return G


def equidistant(R, center='centroid', spacing=1):
    '''
    Returns an array of coordinates for the vertices of a regular triangular
    tiling (spacing sets the triangle's side) within radius R.
    The coordinate origin is in the centroid of the central triangle.
    '''
    lim = (R/spacing)**2
    h = np.sqrt(3)/2

    if center == 'centroid':
        def iswithin(x, y):
            return x**2 + y**2 <= lim

        Vsector = []

        offset = np.sqrt(3)/3
        i = 0
        repeat = True
        # this loop fills a 120° sector
        while True:
            # x0 = (3*i + 2)*h/3
            x0 = i*h + offset
            if i % 2 == 0 and repeat:
                # add line starting at 0°
                y0 = 0
                repeat = False
            else:
                # add line starting at 60°
                y0 = x0*h*2
                repeat = True
                i += 1
            if iswithin(x0, y0):
                Vsector.append((x0, y0))
                c = 1
                while True:
                    x, y = x0 + c*h, y0 + c/2
                    if iswithin(x, y):
                        Vsector.append((x, y))
                        r = np.sqrt(x**2 + y**2)
                        θ = 2*np.pi/3 - np.arctan2(y, x)
                        Vsector.append((r*np.cos(θ), r*np.sin(θ)))
                    else:
                        break
                    c += 1
            else:
                if not repeat:
                    break
        # replicate the 120° sector created to fill the circle
        Vsector = np.array(Vsector)
        r = np.hypot(*Vsector.T)
        θ = np.arctan2(*Vsector.T[::-1])
        cos_sin = tuple(np.c_[np.cos(θ + β), np.sin(θ + β)]
                        for β in (2*np.pi/3, 4*np.pi/3))
        output = np.r_[tuple((Vsector,) +
                             tuple(cs*r[:, np.newaxis] for cs in cos_sin))]

    elif center == 'vertex':
        def addupper(x, y):
            X, Y = (x + 0.5, y + h)
            if X**2 + Y**2 <= lim:
                yield X, Y
                yield from addupper(X, Y)

        def addlower(x, y):
            X, Y = (x + 1, y)
            if X**2 + Y**2 <= lim:
                yield X, Y
                yield from addlower(X, Y)

        def addbranches(x, y):
            yield from addlower(x, y)
            X, Y = (x + 1.5, y + h)
            if X**2 + Y**2 <= lim:
                yield X, Y
                yield from addbranches(X, Y)
            yield from addupper(x, y)

        firstbranch = (1.5, h)
        Vsector = np.array(tuple(addlower(0, 0)) +
                           (firstbranch,) +
                           tuple(addbranches(*firstbranch)))

        # replicate the 60° sector created to fill the circle
        Vsector = np.array(Vsector)
        r = np.hypot(*Vsector.T)
        θ = np.arctan2(*Vsector.T[::-1])
        cos_sin = tuple(np.c_[np.cos(θ + β), np.sin(θ + β)]
                        for β in np.pi/3*np.arange(1, 6))
        output = np.r_[tuple((np.zeros((1, 2), dtype=float),) + (Vsector,) +
                       tuple(cs*r[:, np.newaxis] for cs in cos_sin))]
    else:
        print('Unknown option for <center>:', center)
        return None
    return spacing*output
