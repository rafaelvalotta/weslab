# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import subprocess
import os
from collections.abc import Sequence
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from wesl.optimizer.interarray.interarraylib import cost, calcload, make_graph_metrics, NodeStr
from matplotlib import animation
import numpy as np
# from numpngw import AnimatedPNGWriter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#from pygifsicle import gifsicle


class LayoutPlotter():

    edge_color = 'crimson'
    root_color = 'lawngreen'
    node_tag = 'label'
    colors = plt.cm.get_cmap('tab20', 20).colors

    def __init__(self, G_base, ax=None, dpi=None, node_tag='label'):
        # fail early if graph does not have a log
        loglist = G_base.graph['log']
        if 'has_loads' not in G_base.graph:
            calcload(G_base)

        if dpi is None:
            dpi = plt.rcParams['figure.dpi']
        # self.mm = mm = 25.4*dpi
        self.node_tag = node_tag
        self.G_base = G_base
        self.root_node_size = 28 if self.node_tag is not None else 18
        self.node_size = 70 if self.node_tag is not None else 18

        log = [(0, [['nop', None]])]
        iprev = 0
        for i, *entry in loglist:
            # print(i)
            if i != iprev:
                out = [entry]
                iprev = i
                log.append((i, out))
            else:
                out.append(entry)

        log.append((i, [['end', None]]))
        self.log = log

        VertexC = G_base.graph['VertexC']
        self.VertexC = VertexC
        M = G_base.graph['M']
        self.M = M
        N = G_base.number_of_nodes() - M
        self.N = N
        self.fnT = G_base.graph.get('fnT')
        pos = dict(zip(range(N), VertexC[:N]))
        # pos |= dict(zip(range(-M, 0), VertexC[-M:]))
        pos.update(dict(zip(range(-M, 0), VertexC[-M:])) )

        D = G_base.graph.get('D')
        if D is not None:
            N -= D
        self.pos = pos

        G = nx.Graph(name=G_base.name, M=M, VertexC=VertexC)
        G.add_nodes_from(G_base.nodes(data=True))
        # make star graph
        make_graph_metrics(G)
        d2roots = G.graph['d2roots']
        for n in range(N):
            # root = G_base.nodes[n]['root']
            root = G.nodes[n]['root']
            G.add_edge(root, n, length=d2roots[n, root])
        # for node in G.nbunch_iter(range(N, N + D)):
            # G[node]['color'] = 'none'
        self.G = G

        subtrees = G.nodes(data='subtree', default=-1)
        self.node_colors = np.array([self.colors[subtrees[n] %
                                                 len(self.colors)]
                                     for n in range(N)])
        Subtree = defaultdict(list)
        for node, subtreeI in G.nodes(data='subtree'):
            if subtreeI is None or node >= N:
                continue
            Subtree[subtreeI].append(node)
        self.Subtree = Subtree
        self.uncolored = set(Subtree.keys())
        self.DetourNodeA = {}

        if ax is None:
            # limX, limY = 1920/dpi, 1080/dpi
            limX, limY = 1440/dpi, 900/dpi
            lR = limX/limY
            boundary = G_base.graph['boundary']
            XYrange = np.abs(np.amax(boundary, axis=0) - np.amin(boundary, axis=0))
            ratio = XYrange[0]/XYrange[1]
            figsize = (limX, limX/ratio) if ratio > lR else (limY*ratio, limY)
            self.fig = plt.figure(figsize=figsize)
            # self.fig.facecolor = '#1b1c17'
        self.init_plt()

    def init_plt(self):
        G = self.G
        M = self.M
        pos = self.pos
        # ax = self.ax
        ax = self.fig.add_subplot(aspect='equal')
        ax.axis('off')
        # ax.facecolor = '#1b1c17'
        self.ax = ax

        redraw = []
        # draw farm boundary
        # area_polygon = Polygon(self.G_base.graph['boundary'], color='#111111', zorder=0)
        area_polygon = Polygon(self.G_base.graph['boundary'], color='black', zorder=0)
        self.boundaryA = ax.add_patch(area_polygon)
        redraw.append(self.boundaryA)
        ax.update_datalim(area_polygon.get_xy())
        ax.autoscale()
        ax.set_aspect('equal')

        # draw root nodes
        roots = range(-M, 0)
        RootL = {r: G.nodes[r]['label'] for r in roots[::-1]}
        redraw.append(nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=roots, node_color=self.root_color,
            node_size=self.root_node_size))

        # draw regular nodes, one subtree at a time
        Subtree = self.Subtree
        SubtreeA = np.empty((len(Subtree)), dtype=object)
        for subtreeI, nodes in Subtree.items():
            SubtreeA[subtreeI] = nx.draw_networkx_nodes(
                G, pos, ax=ax, nodelist=Subtree[subtreeI],  node_color=[self.colors[-1]],
                node_size=self.node_size)
            redraw.append(SubtreeA[subtreeI])
        self.SubtreeA = SubtreeA

        # draw labels
        font_size = {'load': 8, 'label': 6}
        node_tag = self.node_tag
        if node_tag is not None:
            if node_tag == 'load' and 'has_loads' not in G.graph:
                node_tag = 'label'
            labels = nx.get_node_attributes(G, node_tag)
            for root in roots:
                if root in labels:
                    labels.pop(root)
            self.labelsA = nx.draw_networkx_labels(
                G, pos, ax=ax, font_size=font_size[node_tag],
                labels=labels)
            redraw.extend(self.labelsA.values())
        # root nodes' labels
        redraw.extend(nx.draw_networkx_labels(G, pos, ax=ax, font_size=5,
                                              labels=RootL).values())
        redraw.append(self.draw_edges())

        # create text element for iteration number
        self.iteration = ax.text(0.01, 0.99, 'i =   0', ha='left', va='top',
                                 transform=ax.transAxes,
                                 fontname='DejaVu Sans Mono',
                                 color='white')
                                 # transform=ax.transAxes, fontname='Inconsolata')
                                 # transform=ax.transAxes, fontname='Iosevka')
        redraw.append(self.iteration)
        # create text element for total length
        self.length = ax.text(0.99, 0.99, f'{G.size(weight="length"):.0f} m',
                              ha='right', va='top', transform=ax.transAxes,
                              fontname='DejaVu Sans Mono',
                              color='white')
                              # fontname='Inconsolata')
                              # fontname='Iosevka')
        redraw.append(self.length)
        return redraw

    def draw_edges(self):
        G = self.G
        edge_colors = [color for u, v, color in
                       G.edges(data='color', default=self.edge_color)]
        edge_style = [style for s, t, style in
                      G.edges(data='style', default='solid')]
        edgesA = nx.draw_networkx_edges(
            G, self.pos, ax=self.ax, edge_color=edge_colors,
            style=edge_style)
        self.prevEdgesA = edgesA
        return edgesA

    def update(self, step):
        redraw = []
        n2s = NodeStr(self.fnT, self.N)
        detourprop = dict(style='dashed', color='yellow')
        G = self.G
        pos = self.pos
        VertexC = self.VertexC
        # TODO: clear highlighting from previous iteration

        i, operations = step
        self.iteration.set_text('i = ' + f'{i:d}'.rjust(3, ' '))
        redraw.append(self.iteration)
        for oper, args in operations:
            if oper == 'addE':
                # if args in self.G_base.edges:
                #     length = self.G_base[args]['length']
                # else:
                #     u, v = args
                #     length = np.hypot(*(VertexC[u] - VertexC[v]).T)
                G.add_edge(*args, length=self.G_base.edges[args]['length'])
            elif oper == 'remE':
                G.remove_edge(*args)
            elif oper == 'addDE':
                s, t, s_, t_ = args
                length = np.hypot(*(VertexC[s_] - VertexC[t_]).T)
                G.add_edge(s, t, **detourprop, length=length)
            elif oper == 'addDN':
                t_, new = args
                G.add_node(new)
                pos[new] = VertexC[t_]
                self.DetourNodeA[new] = nx.draw_networkx_nodes(
                    G, pos, ax=self.ax, nodelist=[new], alpha=0.4,
                    edgecolors='orange', node_color='none', node_size=150)
                redraw.append(self.DetourNodeA[new])
            elif oper == 'movDN':
                hook, corner, hook_, corner_ = args
                root = self.G_base.nodes[hook]['root']
                pos[corner] = VertexC[corner_]
                if corner > 0:
                    self.DetourNodeA[corner].remove()
                    redraw.append(self.DetourNodeA[corner])
                    self.DetourNodeA[corner] = nx.draw_networkx_nodes(
                        G, pos, ax=self.ax, nodelist=[corner], alpha=0.4,
                        edgecolors='orange', node_color='none', node_size=150)
                    redraw.append(self.DetourNodeA[corner])
                hook2cornerL = np.hypot(*(VertexC[hook_] -
                                          VertexC[corner_]).T)
                # corner2rootL = self.G_base.graph['d2roots'][t_, root]
                G.edges[hook, corner]['length'] = hook2cornerL
                # G.edges[blocked, root]['length'] = corner2rootL
            elif oper == 'finalG':
                # TODO: this coloring of the subtree is misleading for the
                # cases where the gate is marked as final before the full
                # capacity is reached, causing nodes still not part of the
                # final subtree to receive its color too early
                gate, root = args
                subtreeI = G.nodes[gate]['subtree']
                self.SubtreeA[subtreeI].remove()
                nodes = self.Subtree[subtreeI]
                self.SubtreeA[subtreeI] = nx.draw_networkx_nodes(
                    G, pos, ax=self.ax, node_size=self.node_size,
                    nodelist=nodes, node_color=self.node_colors[nodes])
                redraw.append(self.SubtreeA[subtreeI])
                # print(i, self.uncolored, f' - {{{subtreeI}}}')
                # self.uncolored.remove(subtreeI)
                self.uncolored.discard(subtreeI)
            elif oper == 'remN':
                G.remove_node(args)
                self.DetourNodeA[args].remove()
                redraw.append(self.DetourNodeA[args])
                del self.DetourNodeA[args]
            elif oper == 'end':
                # color the remaining subtrees
                for subtreeI in self.uncolored:
                    self.SubtreeA[subtreeI].remove()
                    nodes = self.Subtree[subtreeI]
                    self.SubtreeA[subtreeI] = nx.draw_networkx_nodes(
                        G, pos, ax=self.ax, node_size=self.node_size,
                        nodelist=nodes, node_color=self.node_colors[nodes])
                    redraw.append(self.SubtreeA[subtreeI])
                # make text bold to mark the last frame
                self.length.set_color('yellow')
                self.iteration.set_color('yellow')
            elif oper == 'nop':
                pass
            else:
                print('Unknown operation:', oper)

        self.length.set_text(f'{G.size(weight="length"):.0f} m')
        redraw.append(self.length)
        self.prevEdgesA.remove()
        self.prevEdgesA.set_visible(False)
        redraw.append(self.prevEdgesA)
        redraw.append(self.draw_edges())
        return redraw


def animate(G, interval=250, blit=True, workpath='./tmp/', node_tag='label',
            savepath='./videos/', remove_apng=True, use_apng2gif=False):
    # old_dpi = plt.rcParams['figure.dpi']
    # dpi = plt.rcParams['figure.dpi'] = 192
    # layplt = LayoutPlotter(G, dpi=dpi)
    layplt = LayoutPlotter(G, dpi=192, node_tag=node_tag)
    savefig_kwargs = {'facecolor': '#1b1c17'}
    anim = animation.FuncAnimation(
        layplt.fig, layplt.update, frames=layplt.log, interval=interval,
        blit=blit)
    if use_apng2gif:
        print('apng2gif is disabled in the source code.')
        # fname = f'{G.name}_{G.graph["edges_created_by"]}_' \
                # f'{G.graph["capacity"]}.apng'
        # writer = AnimatedPNGWriter(fps=1000/interval)
        # anim.save(workpath + fname, writer=writer,
                  # savefig_kwargs=savefig_kwargs)
        # subprocess.run(['apng2gif', workpath + fname, savepath + fname[:-4] + 'gif'])
        # if remove_apng:
            # os.remove(workpath + fname)
    else:
        fname = f'{G.name}_{G.graph["edges_created_by"]}_' \
                f'{G.graph["capacity"]}.gif'
        writer = animation.PillowWriter(fps=1000/interval)
        anim.save(savepath + fname, writer=writer,
                  savefig_kwargs=savefig_kwargs)
        gifsicle(sources=savepath + fname, options=['--optimize=3'])
    # plt.rcParams['figure.dpi'] = old_dpi
    return fname


def gplot(G, ax=None, node_tag='load', edge_exemption=False, figlims=(5, 6)):
    '''NetworkX graph plotting function.
    `node_tag` in [None, 'load', 'label']
    (or other key in nodes's dict)'''
    figsize = plt.rcParams['figure.figsize']
    dark = plt.rcParams['figure.facecolor'] != 'white'

    if ax is None and not dark:
        limX, limY = figlims
        r = limX/limY
        boundary = G.graph['boundary']
        XYrange = np.abs(np.amax(boundary, axis=0) - np.amin(boundary, axis=0))
        d = XYrange[0]/XYrange[1]
        if d < r:
            figsize = (limY*d, limY)
        else:
            figsize = (limX, limX/d)

    root_size = 28
    detour_size = 150 if node_tag is not None else 80
    node_size = 60 if node_tag is not None else 28

    type2color = {}
    type2style = {}
    type2style['detour'] = 'dashed'
    type2style['unspecified'] = 'solid'
    if dark:
        scalebar = False
        type2color['unspecified'] = 'crimson'
        type2color['detour'] = 'darkorange'
        root_color = 'lawngreen'
        node_edge = 'none'
        detour_ring = 'orange'
        polygon_edge = 'none'
        polygon_face = '#111111'
    else:
        scalebar = True
        type2color['unspecified'] = 'firebrick'
        type2color['detour'] = 'royalblue'
        root_color = 'black' if node_tag is None else 'yellow'
        node_edge = 'black'
        detour_ring = 'deepskyblue'
        polygon_edge = '#444444'
        polygon_face = 'whitesmoke'

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    VertexC = G.graph['VertexC']
    M = G.graph['M']
    N = G.number_of_nodes() - M
    D = G.graph.get('D')
    if D is not None:
        N -= D
        fnT = G.graph.get('fnT')
        if fnT is not None:
            DetourC = VertexC[fnT[range(N, N + D)]]
        else:
            # TODO: deprecate DetourC
            DetourC = G.graph['DetourC']

    # draw farm boundary
    if 'boundary' in G.graph:
        area_polygon = Polygon(G.graph['boundary'], zorder=0, linestyle='--',
                               facecolor=polygon_face, edgecolor=polygon_edge,
                               linewidth=0.3)
        ax.add_patch(area_polygon)
        ax.update_datalim(area_polygon.get_xy())
        ax.autoscale()
    ax.set_aspect('equal')
    # setup
    roots = range(-M, 0)
    pos = dict(zip(range(N), VertexC[:N]))
    # pos |= dict(zip(roots, VertexC[-M:]))

    pos.update(dict(zip(roots, VertexC[-M:])))


    if D is not None:
        detour = range(N, N + D)
        # pos |= dict(zip(detour, DetourC))
        pos.update(dict(zip(detour, DetourC)))


    RootL = {r: G.nodes[r]['label'] for r in roots[::-1]}

    colors = plt.cm.get_cmap('tab20', 20).colors
    # default value for subtree (i.e. color for unconnected nodes)
    # is the last color of the tab20 colormap (i.e. 19)
    subtrees = G.nodes(data='subtree', default=19)
    node_colors = [colors[subtrees[n] % len(colors)] for n in range(N)]

    # draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=type2color['unspecified'],
                           style=type2style['unspecified'], label='direct',
                           edgelist=[(u, v) for u, v, t in G.edges.data('type')
                                     if t is None])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=type2color['detour'],
                           style=type2style['detour'], label='detour',
                           edgelist=[(u, v) for u, v, t in G.edges.data('type')
                                     if t == 'detour'])

    # draw nodes
    if D is not None:
        # draw circunferences around nodes that have Detour clones
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=detour, alpha=0.4,
                               edgecolors=detour_ring, node_color='none',
                               node_size=detour_size,
                               label='corner')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=roots, linewidths=0.2,
                           node_color=root_color, edgecolors=node_edge,
                           node_size=root_size,
                           label='OSS')
    nx.draw_networkx_nodes(G, pos, nodelist=range(N), edgecolors=node_edge,
                           ax=ax, node_color=node_colors, node_size=node_size,
                           linewidths=0.2, label='WTG')

    # draw labels
    font_size = {'load': 7, 'label': 5}
    if node_tag is not None:
        if node_tag == 'load' and 'has_loads' not in G.graph:
            node_tag = 'label'
        labels = nx.get_node_attributes(G, node_tag)
        for root in roots:
            if root in labels:
                labels.pop(root)
        if D is not None:
            for det in detour:
                if det in labels:
                    labels.pop(det)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size[node_tag],
                                labels=labels)
    # root nodes' labels
    if node_tag is not None:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=5,
                                labels=RootL)
    else:
        pass
        # nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=roots, alpha=0.2,
        #                        node_color=root_color, node_size=150)

    if scalebar:
        bar = AnchoredSizeBar(ax.transData, 1000, '1 km', 'lower right', frameon=False)
        ax.add_artist(bar)

    if 'capacity' in G.graph:
        legend = [f'$\\kappa = {G.graph["capacity"]}$, '
                  f'$N = {N}$']
        feeder_info = [f'$\\phi_{{{rootL}}}$ = {len(G[r])}'
                       for r, rootL in RootL.items()]
        if 'overfed' in G.graph:
            feeder_info = [fi + f' ({100*(overfed - 1):+.0f}%)'
                           for fi, overfed in
                           zip(feeder_info, G.graph['overfed'][::-1])]
        legend.extend(feeder_info)
        # legend.append(', '.join(feeder_info))
        # for field, sym in (('weight', 'w'), ('length', 'l')):
        for field, sym in (('length', ''),):
            weight = field if all([(field in data)
                                   for _, _, data in G.edges.data()]) else None
            legend.append('Σ{} = {:.0f}'.format(sym, G.size(weight=weight)) +
                          ' m' if field == 'length' else '')
    if ('has_loads' in G.graph) and ('cables' in G.graph):
        legend.append('{:.0f} €'.format(cost(G)))
    if 'capacity' in G.graph:
        infobox = ax.legend([], fontsize=7, title='\n'.join(legend),
                            labelspacing=0)  # ,   loc='upper right',
                            # bbox_to_anchor=(-0.04, 0.80, 1.08, 0))
                            # bbox_to_anchor=(-0.04, 1.03, 1.08, 0))
        plt.setp(infobox.get_title(), multialignment='center')
        # ax.legend(title='\n'.join(legend))
        # legend1 = pyplot.legend(plot_lines[0], ["algo1", "algo2", "algo3"], loc=1)
        # pyplot.legend([l[0] for l in plot_lines], parameters, loc=4)
        # ax.add_artist(legstrip)
    if not dark:
        legstrip = ax.legend(ncol=8, fontsize=6, loc='lower center',
                             frameon=False, bbox_to_anchor=(0.5, -0.07),
                             columnspacing=1, handletextpad=0.3)
        if 'capacity' in G.graph:
            ax.add_artist(infobox)
        # infobox = ax.legend([], title='\n'.join(legend))
    return ax


def compare(positional=None, **title2G_dict):
    '''Plot layouts side by side. dict keys are inserted in the title.'''
    if positional is not None:
        if isinstance(positional, Sequence):
            # title2G_dict |= {chr(i): val for i, val in
            #                  enumerate(positional, start=ord('A'))}
            
            title2G_dict.update({chr(i): val for i, val in
                             enumerate(positional, start=ord('A'))} )

        else:
            title2G_dict[''] = positional
    fig, axes = plt.subplots(1, len(title2G_dict), squeeze=False)
    for ax, (title, G) in zip(axes.ravel(), title2G_dict.items()):
        gplot(G, ax=ax, node_tag=None)
        ax.set_title(f'{title} – {G.graph["name"]} '
                     f'({G.graph["edges_created_by"]})')
