import os
import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

episode = 1000000
trajectory = []
chosen_episode = -1
for k in range(episode, 0, -10):
    if os.path.exists("results/trajectory_{}.pkl".format(k)):
        print("{}-th episode".format(k))
        chosen_episode = episode
        with open("results/reward_history_{}.pkl".format(k), "rb") as f:
            result_history = pickle.load(f)
            # plt.plot(
            #     [i for i in range(len(result_history))],
            #     result_history,
            # )
            # plt.show()
        if result_history[-1] <= -22:
            continue
        with open("results/trajectory_{}.pkl".format(k), "rb") as f:
            data = pickle.load(f)
            for t in range(len(data)):
                trajectory.append(
                    torch.tensor(data[t])
                    .reshape(4, -1)[:, -2:]
                    .reshape(1, -1)
                    .squeeze()
                    .tolist()
                )
        break

fig = plt.figure()
ax = fig.add_subplot(111, aspect=1)

color = ["purple", "blue", "green", "orange"]


def update(f):
    ax.cla()
    # ax.grid()
    ax.axis([-1, 6, -1, 7])
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # ゴールを描画する
    goals = [
        (4.5, 5.5),
        (4.5, -0.5),
        (-0.5, 5.5),
        (-0.5, -0.5),
    ]
    goals_color = ["purple", "blue", "green", "orange"]
    goals_patches = [
        patches.Rectangle(
            xy=goals[i][0:2],
            width=1,
            height=1,
            fc=goals_color[i],
            fill=True,
        )
        for i in range(len(goals))
    ]

    for goal_patch in goals_patches:
        ax.add_patch(goal_patch)

    # 黒いエリアを描画する
    black = [
        (-0.5, 1.5),
        (1.5, 1.5),
        (4.5, 1.5),
    ]
    black_shape = [(1, 3), (2, 3), (1, 3)]
    black_block = [
        patches.Rectangle(
            xy=black[i][0:2],
            width=black_shape[i][0],
            height=black_shape[i][1],
            fc="black",
            fill=True,
        )
        for i in range(len(black))
    ]
    for i in range(len(black)):
        ax.add_patch(black_block[i])

    for i in range(9):
        a = -0.5 + i
        ax.vlines(a, -0.5, 7.5, colors="black", linestyles="dashed")
        ax.hlines(a, -0.5, 7.5, colors="black", linestyles="dashed")

    for i in range(4):
        ax.plot(f[i * 2], f[2 * i + 1], "o", c=color[i])


anim = FuncAnimation(fig, update, frames=trajectory, interval=500)
plt.show()
anim.save("trajectory_{}.gif".format(chosen_episode), writer="imagemagick")
plt.close()
