# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
from math import isclose
import operator
from collections import defaultdict
from itertools import product
import functools
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from wesl.optimizer.interarray.interarraylib import NodeTagger, NodeStr

F = NodeTagger()


def any_pairs_opposite_edge(NodesC, uC, vC, margin=0):
    '''Returns True if any two of `NodesC` are on opposite
    sides of the edge (`uC`, `vC`).
    '''
    maxidx = len(NodesC) - 1
    if maxidx <= 0:
        return False
    refC = NodesC[0]
    i = 1
    while point_d2line(refC, uC, vC) <= margin:
        # ref node is approx. overlapping the edge: get the next one
        refC = NodesC[i]
        i += 1
        if i > maxidx:
            return False

    for cmpC in NodesC[i:]:
        if point_d2line(cmpC, uC, vC) <= margin:
            # cmp node is approx. overlapping the edge: skip
            continue
        if not is_same_side(uC, vC, refC, cmpC,
                            touch_is_cross=False):
            return True
    return False


def rotate(coords, angle):
    '''rotates `coords` (numpy array N×2) by `angle` (degrees)'''
    rotation = np.deg2rad(angle)
    c, s = np.cos(rotation), np.sin(rotation)
    return np.dot(coords, np.array([[c, s], [-s, c]]))


def point_d2line(p, u, v):
    '''calculates the distance from point `p` to the line defined by points `u` and `v`'''
    x0, y0 = p
    x1, y1 = u
    x2, y2 = v
    return (abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)) /
            math.sqrt((x2 - x1)**2 + (y2 - y1)**2))


def angle_numpy(a, pivot, b):
    '''a, pivot, b are coordinate pairs
    returns angle a-root-b (radians)
    - angle is within ±π (shortest arc from a to b around pivot)
    - positive direction is counter-clockwise'''
    A = a - pivot
    B = b - pivot
    # dot_prod = np.dot(A, B) if len(A) >= len(B) else np.dot(B, A)
    dot_prod = A @ B.T  # if len(A) >= len(B) else np.dot(B, A)
    # return np.arctan2(np.cross(A, B), np.dot(A, B))
    return np.arctan2(np.cross(A, B), dot_prod)


def angle(a, pivot, b):
    '''`a`, `pivot`, `b` are coordinate pairs
    returns angle a-root-b (radians)
    - angle is within ±π (shortest arc from a to b around pivot)
    - positive direction is counter-clockwise'''
    Ax, Ay = a - pivot
    Bx, By = b - pivot
    # debug and print(VertexC[a], VertexC[b])
    ang = np.arctan2(Ax*By - Ay*Bx, Ax*Bx + Ay*By)
    # debug and print(f'{ang*180/np.pi:.1f}')
    return ang


def is_bb_overlapping(uv, wy):
    ''' checks if there is an overlap in the bounding boxes of `uv` and `wy`
    (per row)
    `uv` and `wy` have shape N×2, '''
    pass


def is_crossing_numpy(u, v, w, y):
    '''checks if (u, v) crosses (w, y);
    returns ¿? in case of superposition'''

    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphics Gems III (http://www.graphicsgems.org/)
    # license: https://github.com/erich666/GraphicsGems/blob/master/LICENSE.md

    A = v - u
    B = w - y

    # bounding box check
    for i in (0, 1):  # X and Y
        lo, hi = (v[i], u[i]) if A[i] < 0 else (u[i], v[i])
        if B[i] > 0:
            if hi < y[i] or w[i] < lo:
                return False
        else:
            if hi < w[i] or y[i] < lo:
                return False

    C = u - w

    # denominator
    f = np.cross(B, A)
    if f == 0:
        # segments are parallel
        return False

    # alpha and beta numerators
    for num in (np.cross(P, Q) for P, Q in ((C, B), (A, C))):
        if f > 0:
            if num < 0 or num > f:
                return False
        else:
            if num > 0 or num < f:
                return False

    # code to calculate intersection coordinates omitted
    # segments do cross
    return True


def is_crossing(u, v, w, y, touch_is_cross=True):
    '''checks if (u, v) crosses (w, y);
    returns ¿? in case of superposition
    choices for `less`:
    -> operator.lt counts touching as crossing
    -> operator.le does not count touching as crossing
    '''
    less = operator.lt if touch_is_cross else operator.le

    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphic Gems III

    A = v - u
    B = w - y

    # bounding box check
    for i in (0, 1):  # X and Y
        lo, hi = (v[i], u[i]) if A[i] < 0 else (u[i], v[i])
        if B[i] > 0:
            if hi < y[i] or w[i] < lo:
                return False
        else:
            if hi < w[i] or y[i] < lo:
                return False

    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C = u - w

    # denominator
    # print(Ax, Ay, Bx, By)
    f = Bx*Ay - By*Ax
    # print('how close: ', f)
    # TODO: arbitrary threshold
    if isclose(f, 0, abs_tol=1e-3):
        # segments are parallel
        return False

    # alpha and beta numerators
    for num in (Px*Qy - Py*Qx for (Px, Py), (Qx, Qy) in ((C, B), (A, C))):
        if f > 0:
            if less(num, 0) or less(f, num):
                return False
        else:
            if less(0, num) or less(num, f):
                return False

    # code to calculate intersection coordinates omitted
    # segments do cross
    return True


def is_bunch_split_by_corner(bunch, a, o, b, margin=1e-3):
    '''`bunch` is a numpy array of points (N×2)
    the points `a`-`o`-`b` define a corner'''
    AngleA = angle_numpy(a, o, bunch)
    AngleB = angle_numpy(b, o, bunch)
    # print('AngleA', AngleA, 'AngleB', AngleB)
    # keep only those that don't fall over the angle-defining lines
    keep = ~np.logical_or(np.isclose(AngleA, 0, atol=margin),
                          np.isclose(AngleB, 0, atol=margin))
    angleAB = angle(a, o, b)
    angAB = angleAB > 0
    inA = AngleA > 0 if angAB else AngleA < 0
    inB = AngleB > 0 if ~angAB else AngleB < 0
    # print(angleAB, keep, inA, inB)
    inside = np.logical_and(keep, np.logical_and(inA, inB))
    outside = np.logical_and(keep, np.logical_or(~inA, ~inB))
    split = any(inside) and any(outside)
    return split, np.flatnonzero(inside), np.flatnonzero(outside)


def is_quadrilateral_convex(u, v, w, y):
    '''(u, v) is the edge;
    w, y are the opposing nodes;
    returns False also if it is a triangle
    only works if w and y are not on the same side of line u-v'''
    # uw × uy
    uwuy = np.cross(w - u, y - u)
    # vy × vw
    vyvw = np.cross(y - v, w - v)
    return uwuy*vyvw > 100  # TODO: ARBITRARY - depends on scale


def is_same_side(L1, L2, A, B, touch_is_cross=True):
    '''check if points A an B are on the same side
    of the line defined by points L1 and L2'''
    # greater = operator.gt if touch_is_cross else operator.ge
    greater = operator.ge if touch_is_cross else operator.gt
    # print(L1, L2, A, B)
    (Ax, Ay), (Bx, By), (L1x, L1y), (L2x, L2y) = (A, B, L1, L2)
    denom = (L1x - L2x)
    # test to avoid division by zero
    if denom:
        a = -(L1y - L2y)/denom
        c = -a*L1x - L1y
        num = a*Ax + Ay + c
        den = a*Bx + By + c
        discriminator = num*den
    else:
        # this means the line is vertical (L1x = L2x)
        # which makes the test simpler
        discriminator = (Ax - L1x)*(Bx - L1x)
    return greater(discriminator, 0)


def is_blocking(root, u, v, w, y):
    # w and y are necessarily on opposite sides of uv
    # (because of Delaunay – see the triangles construction)
    # hence, if (root, y) are on the same side, (w, root) are not
    return (is_quadrilateral_convex(u, v, w, root)
            if is_same_side(u, v, root, y)
            else is_quadrilateral_convex(u, v, root, y))


def apply_edge_exemptions(G, allow_edge_deletion=True):
    '''exemption is used by weighting functions that take
    into account the angular sector blocked by each edge w.r.t.
    the closest root node
    '''
    E_hull = G.graph['E_hull']
    N_hull = G.graph['N_hull']
    N_inner = set(G.nodes) - N_hull
    M = G.graph['M']
    # N = G.number_of_nodes() - M
    VertexC = G.graph['VertexC']
    # roots = range(N, N + M)
    roots = range(-M, 0)
    triangles = G.graph['triangles']
    angles = G.graph['angles']

    # set hull edges as exempted
    for edge in E_hull:
        G.edges[edge]['exempted'] = True

    # expanded E_hull to contain edges exempted from blockage penalty
    # (edges that do not block line from nodes to root)
    E_hull_exp = E_hull.copy()

    # check if edges touching the hull should be exempted from blockage penalty
    for n_hull in N_hull:
        for n_inner in (N_inner & set([v for u, v in G.edges(n_hull)])):
            uv = frozenset((n_hull, n_inner))
            u, v = uv
            opposites = triangles[uv]
            if len(opposites) == 2:
                w, y = triangles[uv]
                rootC = VertexC[G.edges[u, v]['root']]
                uvwyC = tuple((VertexC[n] for n in (*uv, w, y)))
                if not is_blocking(rootC, *uvwyC):
                    E_hull_exp.add(uv)
                    G.edges[uv]['exempted'] = True

    # calculate blockage arc for each edge
    zeros = np.full((M,), 0.)
    for u, v, d in list(G.edges(data=True)):
        if (frozenset((u, v)) in E_hull_exp) or (u in roots) or (v in roots):
            angdiff = zeros
        else:
            # angdiff = (angles[:, u] - angles[:, v]) % (2*np.pi)
            # angdiff = abs(angles[:, u] - angles[:, v])
            angdiff = abs(angles[u] - angles[v])
        arc = np.empty((M,), dtype=float)
        for i in range(M):  # TODO: vectorize this loop
            arc[i] = angdiff[i] if angdiff[i] < np.pi else 2*np.pi - angdiff[i]
        d['arc'] = arc
        # if arc is π/2 or more, remove the edge (it's shorter to go to root)
        if allow_edge_deletion and any(arc >= np.pi/2):
            G.remove_edge(u, v)
            print('angles', arc, 'removing «',
                  '–'.join([F[n] for n in (u, v)]), '»')


def edge_crossings(u, v, G, triangles, triangles_exp):
    '''
    This only works for subgraphs of a delaunay base with add_diagonals=True.
    Other edges (e.g. gate edges) are not implemented here.
    '''
    uv = frozenset((u, v))
    crossings = []
    # n2s = NodeStr(G.graph['fnT'], G.graph['N'])
    if uv in triangles:
        # <(u, v) is a Delaunay edge>
        st = triangles[uv]
        if (len(st) > 1 and st in triangles_exp
                and triangles_exp[st] == uv
                and tuple(st) in G.edges):
            return([tuple(st)])
    elif uv in triangles_exp:
        # <(u, v) is an expanded Delaunay edge>
        s, t = triangles_exp[uv]
        if (s, t) in G.edges:
            crossings.append((s, t))
        for a_b in ((u, s), (u, t), (s, v), (t, v)):
            ab = frozenset(a_b)
            cd = triangles.get(ab)
            if cd is None:
                continue
            if (cd in triangles_exp
                    and triangles_exp[cd] == ab
                    and tuple(cd) in G.edges):
                crossings.append(tuple(cd))
    return crossings


def delaunay(G_base, add_diagonals=True, debug=False, MIN_TRI_AREA=1500.,
             threshold=2.15, **qhull_options):
    '''Creates a networkx graph from the Delaunay triangulation
    of the point in coordinates. The weights of each edge is the
    euclidean distance between its vertices.'''
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    M = G_base.graph['M']
    N = G_base.number_of_nodes() - M
    VertexC = G_base.graph['VertexC']
    RootC = VertexC[N:]

    tri = Delaunay(VertexC, **qhull_options)

    triangles = defaultdict(list)
    triangles_exp = defaultdict(list)

    # from the triangles, create graph edges
    for vertices in tri.simplices:
        A, B, C = (V if V < N else V - N - M for V in vertices)
        for V1, V2, V3 in ((A, B, C), (A, C, B), (C, B, A)):
            pair = frozenset((V1, V2))
            triangles[pair].append(V3)
        nx.add_path(G, (A, B, C, A))

    # make <triangles>'s values frozenset instead of list
    triangles = {k: frozenset(v) for k, v in triangles.items()}

    # find out the edges that form the convex hull
    E_hull = set([frozenset((X, Y))
                  for X, Y in ((V if V < N else V - N - M for V in edge) for
                               edge in tri.convex_hull)])
    N_hull = set(functools.reduce(operator.or_, E_hull))
    N_inner = set(G.nodes) - N_hull

    def hull_edge_is_overlapping(edge):
        u, v = edge
        for nb in (N_inner & set(G[u])):
            uC, vC, nbC = VertexC[(u, v, nb), ]
            discriminator = abs(np.cross(uC - nbC, vC - nbC))
            if discriminator < 30e4:  # TODO: ARBITRARY - depends on scale
                Ax, Ay = uC
                Bx, By = vC
                Cx, Cy = nbC
                # TODO: move the test below to its own function
                # t is the normalized projection of C over AB
                t = ((Cx-Ax)*(Bx-Ax)+(Cy-Ay)*(By-Ay))/((Bx-Ax)**2+(By-Ay)**2)
                # if nb is not between u and v, go to next nb
                if t <= 0 or t >= 1:
                    continue
                E_hull.add(frozenset((u, nb)))
                E_hull.add(frozenset((nb, v)))
                uv = frozenset((u, v))
                E_hull.remove(uv)
                if uv in triangles:
                    triangles.pop(uv)
                if (u, v) in G.edges:
                    G.remove_edge(u, v)
                    # print('overlapping', [F[n] for n in uv])
                N_hull.add(nb)
                N_inner.remove(nb)
                hull_edge_is_overlapping((nb, v))
                return True
        return False

    # contract the convex hull to add nodes that are almost part of the hull
    for edge in tuple(E_hull):
        hull_edge_is_overlapping(edge)

    # clean up edges that overlap the ones on the convex hull
    for u in N_hull:
        for v in list(G[u].keys()):
            if (v in N_hull) and (frozenset((u, v)) not in E_hull):
                uv = frozenset((u, v))
                if uv in triangles:
                    nn = VertexC[tuple(triangles[uv]), ]
                    discriminator = abs(np.cross(VertexC[u] - nn,
                                                 VertexC[v] - nn))
                    # TODO: threshold is arbitary, depends on scale
                    if all(discriminator > MIN_TRI_AREA):
                        continue
                    triangles.pop(uv)
                G.remove_edge(u, v)
                # print('clean up:', F[u], F[v])

    # save the convex hull node set
    G.graph['N_hull'] = N_hull
    # save the convex hull edge set
    G.graph['E_hull'] = E_hull

    # sqrt(3) was not high enough to get all diagonals in the g.tess farm
    # threshold = np.sqrt(3)
    # threshold = 2  # value that augments tess
    # threshold = 2.15  # value that augments horns
    for u, v, edgeD in list(G.edges(data=True)):
        # assign the edge to the root closest to the edge's middle point
        edgeD['root'] = -M + np.argmin(
            cdist(((VertexC[u] + VertexC[v])/2)[np.newaxis, :], RootC))
        # add edges' lengths
        u2v = np.hypot(*(VertexC[u] - VertexC[v]).T)
        edgeD['length'] = u2v
        if add_diagonals:
            # add additional diagonals
            uv = frozenset((u, v))
            opposites = triangles[uv]
            if len(opposites) == 2:
                wy = frozenset(opposites)
                if wy in triangles:
                    continue
                w, y = opposites
                uC, vC, wC, yC = VertexC[[u, v, w, y]]
                w2y = np.hypot(*(wC - yC).T)
                if ((not ((w in N_hull) and (y in N_hull))) and
                        (w2y < u2v*threshold) and
                        is_quadrilateral_convex(uC, vC, wC, yC) and
                        (abs(np.cross(wC - uC, yC - uC)) > MIN_TRI_AREA) and
                        (abs(np.cross(wC - vC, yC - vC)) > MIN_TRI_AREA)):
                    wy_root = -M + np.argmin(
                        cdist(((VertexC[w] + VertexC[y])/2)[np.newaxis, :],
                              RootC))
                    G.add_edge(w, y, length=w2y, root=wy_root)
                    triangles_exp[wy] = uv

    G.graph['triangles'] = triangles
    G.graph['triangles_exp'] = triangles_exp

    return G


def get_crossings_map(Edge, VertexC, prune=True):
    crossings = defaultdict(list)
    for i, A in enumerate(Edge[:-1]):
        u, v = A
        uC, vC = VertexC[A]
        for B in Edge[i+1:]:
            s, t = B
            if s == u or t == u or s == v or t == v:
                # <edges have a common node>
                continue
            sC, tC = VertexC[B]
            if is_crossing(uC, vC, sC, tC):
                crossings[frozenset((*A,))].append((*B,))
                crossings[frozenset((*B,))].append((*A,))
    return crossings


# TODO: test this implementation
def full_graph(G_base, include_roots=False, prune=True, crossings=False):
    '''Creates a networkx graph connecting all non-root nodes to every
    other non-root node. Edges with an arc > pi/2 around root are discarded
    The weight of each edge is the euclidean distance between its vertices.'''
    G = nx.Graph()
    M = G_base.graph['M']
    N = G_base.number_of_nodes() - M
    VertexC = G_base.graph['VertexC']
    NodeC = VertexC[:-M]
    RootC = VertexC[-M:]
    Root = range(-M, 0)
    if include_roots:
        G = nx.complete_graph(N + M)
        nx.relabel_nodes(G, dict(zip(range(N, N + M), Root)),
                         copy=False)
        C = cdist(VertexC, VertexC)
    else:
        G = nx.complete_graph(N)
        C = cdist(NodeC, NodeC)
    EdgeFull = np.array(G.edges())
    if prune:
        # prune edges that cover more than 90° angle from any root
        SrcC = VertexC[EdgeFull[:, 0]]
        DstC = VertexC[EdgeFull[:, 1]]
        mask = np.zeros((EdgeFull.shape[0],), dtype=bool)
        for root in Root:
            rootC = VertexC[root]
            # calculates the dot product of vectors representing the
            # nodes of each edge wrt root; then mark the negative ones
            # (angle > pi/2) on `mask`
            mask |= ((SrcC - rootC)*(DstC - rootC)).sum(axis=1) < 0
        # discard edges that cover an arc > pi/2 seen from any root
        G.remove_edges_from(EdgeFull[mask])
        Edge = EdgeFull[~mask]
    else:
        Edge = EdgeFull
    if crossings:
        # get_crossings_map() takes time and space
        G.graph['crossings'] = get_crossings_map(Edge, VertexC)
    # assign nodes to roots?
    # remove edges between nodes belonging to distinct roots whose length is
    # greater than both d2root
    G.graph.update(G_base.graph)
    nx.set_node_attributes(G, G_base.nodes)
    for u, v, edgeD in G.edges(data=True):
        edgeD['length'] = C[u, v]
        # assign the edge to the root closest to the edge's middle point
        edgeD['root'] = -M + np.argmin(
            cdist(((VertexC[u] + VertexC[v])/2)[np.newaxis, :], RootC))
    return G


# TODO: MARGIN is ARBITRARY - depends on the scale
def check_crossings(G, debug=False, MARGIN=0.1):
    '''Checks for crossings (touch/overlap is not considered crossing).
    This is an independent check on the tree resulting from the heuristic.
    It is not supposed to be used within the heuristic.
    MARGIN is how far an edge can advance across another one and still not be
    considered a crossing.'''
    VertexC = G.graph['VertexC']
    M = G.graph['M']
    N = G.number_of_nodes() - M

    D = G.graph.get('D')
    if D is not None:
        N -= D
        # detournodes = range(N, N + D)
        # G.add_nodes_from(((s, {'type': 'detour'})
        #                   for s in detournodes))
        # clone2prime = G.graph['clone2prime']
        # assert len(clone2prime) == D, \
        #     'len(clone2prime) != D'
        # fnT = np.arange(N + D + M)
        # fnT[N: N + D] = clone2prime
        # DetourC = VertexC[clone2prime].copy()
        fnT = G.graph['fnT']
        AllnodesC = np.vstack((VertexC[:N], VertexC[fnT[N:N + D]], VertexC[-M:]))
    else:
        fnT = np.arange(N + M)
        AllnodesC = VertexC
    roots = range(-M, 0)
    fnT[-M:] = roots
    n2s = NodeStr(fnT, N)

    crossings = []
    pivot_plus_edge = []

    def check_neighbors(neighbors, w, x, pivots):
        '''Neighbors is a bunch of nodes, `pivots` is used only for reporting.
        (`w`, `x`) is the edge to be checked if it splits neighbors apart.
        '''
        maxidx = len(neighbors) - 1
        if maxidx <= 0:
            return
        ref = neighbors[0]
        i = 1
        while point_d2line(*AllnodesC[[ref, w, x]]) < MARGIN:
            # ref node is approx. overlapping the edge: get the next one
            ref = neighbors[i]
            i += 1
            if i > maxidx:
                return

        for n2test in neighbors[i:]:
            if point_d2line(*AllnodesC[[n2test, w, x]]) < MARGIN:
                # cmp node is approx. overlapping the edge: skip
                continue
            # print(F[fnT[w]], F[fnT[x]], F[fnT[ref]], F[fnT[cmp]])
            if not is_same_side(*AllnodesC[[w, x, ref, n2test]],
                                touch_is_cross=False):
                print(f'ERROR <splitting>: edge {n2s(w, x)} crosses '
                      f'{n2s(ref, *pivots, n2test)}')
                # crossings.append(((w,  x), (ref, pivot, cmp)))
                crossings.append(((w,  x), (ref, n2test)))
                return True

    for root in roots:
        # edges = list(nx.edge_dfs(G, source=root))
        edges = list(nx.edge_bfs(G, source=root))
        # outstr = ', '.join([f'«{F[fnT[u]]}–{F[fnT[v]]}»' for u, v in edges])
        # print(outstr)
        potential = []
        for i, (u, v) in enumerate(edges):
            for s, t in edges[(i + 1):]:
                if s == u or s == v:
                    continue
                uvst = np.array((u, v, s, t), dtype=int)
                if is_crossing(*AllnodesC[uvst], touch_is_cross=True):
                    potential.append(uvst)
                    distances = np.fromiter(
                        (point_d2line(*AllnodesC[[p, w, x]])
                         for p, w, x in ((u, s, t),
                                         (v, s, t),
                                         (s, u, v),
                                         (t, u, v))),
                        dtype=float,
                        count=4)
                    # print('distances[' +
                    #       ', '.join((F[fnT[n]] for n in (u, v, s, t))) +
                    #       ']: ', distances)
                    nearmask = distances < MARGIN
                    close_count = sum(nearmask)
                    # print('close_count =', close_count)
                    if close_count == 0:
                        # (u, v) crosses (s, t) away from nodes
                        crossings.append(((u, v), (s, t)))
                        # print(distances)
                        print(f'ERROR <edge-edge>: edge «{F[fnT[u]]}–{F[fnT[v]]}» crosses '
                              f'«{F[fnT[s]]}–{F[fnT[t]]}»')
                    elif close_count == 1:
                        # (u, v) and (s, t) touch node-to-edge
                        pivotI, = np.flatnonzero(nearmask)
                        w, x = (u, v) if pivotI > 1 else (s, t)
                        pivot = uvst[pivotI]
                        neighbors = list(G[pivot])
                        entry = (pivot, w, x)
                        if (entry not in pivot_plus_edge and
                                check_neighbors(neighbors, w, x, (pivot,))):
                            pivot_plus_edge.append(entry)
                    elif close_count == 2:
                        # (u, v) and (s, t) touch node-to-node
                        touch_uv, touch_st = uvst[np.flatnonzero(nearmask)]
                        free_uv, free_st = uvst[np.flatnonzero(~nearmask)]
                        # print(f'touch/free u, v :«{F[fnT[touch_uv]]}–'
                        #       f'{F[fnT[free_uv]]}»; s, t:«{F[fnT[touch_st]]}–'
                        #       f'{F[fnT[free_st]]}»')
                        nb_uv, nb_st = list(G[touch_uv]), list(G[touch_st])
                        # print([F[fnT[n]] for n in nb_uv])
                        # print([F[fnT[n]] for n in nb_st])
                        nbNuv, nbNst = len(nb_uv), len(nb_st)
                        if nbNuv == 1 or nbNst == 1:
                            # <a leaf node with a clone – not a crossing>
                            continue
                        elif nbNuv == 2:
                            crossing = is_bunch_split_by_corner(
                                AllnodesC[nb_st],
                                *AllnodesC[[nb_uv[0], touch_uv, nb_uv[1]]],
                                margin=MARGIN)[0]
                        elif nbNst == 2:
                            crossing = is_bunch_split_by_corner(
                                AllnodesC[nb_uv],
                                *AllnodesC[[nb_st[0], touch_st, nb_st[1]]],
                                margin=MARGIN)[0]
                        else:
                            print('UNEXPECTED case!!! Look into it!')
                            # mark as crossing just to make sure it is noticed
                            crossing = True
                        if crossing:
                            print(f'ERROR <split>: edges «{F[fnT[u]]}–{F[fnT[v]]}» '
                                  f'and «{F[fnT[s]]}–{F[fnT[t]]}» break a bunch '
                                  f'apart at {F[fnT[touch_uv]]}, {F[fnT[touch_st]]}')
                            crossings.append(((u,  v), (s, t)))
                    else:  # close_count > 2:
                        # segments (u, v) and (s, t) are almost parallel
                        # find the two nodes furthest apart
                        pairs = np.array(((u, v), (u, s), (u, t),
                                          (s, t), (v, t), (v, s)))
                        furthest = np.argmax(
                            np.hypot(*(AllnodesC[pairs[:, 0]] -
                                       AllnodesC[pairs[:, 1]]).T))
                        # print('furthest =', furthest)
                        w, x = pairs[furthest]
                        q, r = pairs[furthest - 3]
                        if furthest % 3 == 0:
                            # (q, r) is contained within (w, x)
                            neighbors = list(G[q]) + list(G[r])
                            neighbors.remove(q)
                            neighbors.remove(r)
                            check_neighbors(neighbors, w, x, (q, r))
                        else:
                            # (u, v) partially overlaps (s, t)
                            neighbors_q = list(G[q])
                            neighbors_q.remove(w)
                            check_neighbors(neighbors_q, s, t, (q,))
                            # print(crossings)
                            neighbors_r = list(G[r])
                            neighbors_r.remove(x)
                            check_neighbors(neighbors_r, u, v, (r,))
                            # print(crossings)
                            if neighbors_q and neighbors_r:
                                for a, b in product(neighbors_q, neighbors_r):
                                    if is_same_side(*AllnodesC[[q, r, a, b]]):
                                        print(f'ERROR <partial ovelap>: edge '
                                              f'«{F[fnT[u]]}–{F[fnT[v]]}» '
                                              f'crosses «{F[fnT[s]]}–{F[fnT[t]]}»')
                                        crossings.append(((u,  v), (s, t)))
    debug and potential and print(
        'potential crossings: ' +
        ', '.join([f'«{F[fnT[u]]}–{F[fnT[v]]}» × «{F[fnT[s]]}–{F[fnT[t]]}»'
                   for u, v, s, t in potential]))
    return crossings
