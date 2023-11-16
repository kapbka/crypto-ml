import os
from itertools import cycle
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas import Timestamp


def plot_columns(df: pd.DataFrame, y_names: List[str], size_x: int = 200, size_y: int = 16,
                 text: List[Tuple[Timestamp, str]] = [], annotation_count=4):
    cycol = cycle('bgrcmyk')
    fig, host = plt.subplots(figsize=(size_x, size_y))

    host.set_xlabel('time')
    host.set_ylabel(y_names[0])

    p1, = host.plot(df[y_names[0]], color=next(cycol), label=y_names[0])
    host.yaxis.label.set_color(p1.get_color())
    main_patch = mpatches.Patch(color=p1.get_color(), label=y_names[0])

    position_cnt = 0.0
    color_cnt = 0.0
    patches = [main_patch]

    for col in y_names[1:]:
        color_cnt += 0.3
        position_cnt += 60

        graph = host.twinx()
        graph.set_ylabel(col)
        graph.spines['right'].set_position(('outward', position_cnt))

        if isinstance(col, list):
            for c in col:
                color = next(cycol)
                graph.yaxis.label.set_color(color)
                graph.plot(df[c], color=color, label=c)
                patches.append(mpatches.Patch(color=color, label=c))
        else:
            color = next(cycol)
            graph.yaxis.label.set_color(color)
            graph.plot(df[col], color=color, label=col)
            patches.append(mpatches.Patch(color=color, label=col))

    plt.legend(handles=patches, loc='lower left')

    min_v = df[y_names[-1]].min()
    max_v = df[y_names[-1]].max()

    if text:
        offsets = {i: min_v + (i / annotation_count) * (max_v - min_v) for i in range(annotation_count)}

        count = 0
        for ts, txt in text:
            plt.text(ts, offsets[count % annotation_count], txt, fontsize=10)
            count += 1

    return fig


def interactive_plot(df, columns, height=700, width=800):
    colors = cycle(px.colors.qualitative.Plotly)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[columns[0]],
        name=columns[0]
    ))

    first_color = next(colors)
    props = {
        'xaxis': dict(
            domain=[0.02 * (len(columns)), 1]
        ),
        'yaxis': dict(
            # title=columns[0],
            # titlefont=dict(
            #     color=first_color
            # ),
            tickfont=dict(
                color=first_color
            )
        )
    }

    for idx, c in enumerate(columns[1:]):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[c],
            name=c,
            yaxis=f"y{idx + 2}"
        ))

        next_color = next(colors)
        props.update({
            f'yaxis{idx + 2}': dict(
                # title=c,
                # titlefont=dict(
                #     color=next_color
                # ),
                tickfont=dict(
                    color=next_color
                ),
                anchor="free",
                overlaying="y",
                side="left",
                position=0.02 * (idx + 1)
            )
        })

    # Update layout properties
    fig.update_layout(
        width=width,
        height=height,
        template='plotly_dark',
        **props
    )
    return fig


def plot_to_file(df: pd.DataFrame, bots: List[str], graph_path: str, size_x: int, size_y: int, cols: List[str] = []):
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)

    fig = plot_columns(df, ['close', bots] + cols, size_x=size_x, size_y=size_y)
    fig.savefig(graph_path)
    fig.clear()
    plt.close(fig)
