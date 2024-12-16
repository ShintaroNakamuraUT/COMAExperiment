import copy

import numpy as np
import torch


class Env:
    def __init__(
        self,
    ):
        self.agent_coor = [[0, 0] for _ in range(4)]
        self.time = 1

        self.corridor = [
            [1, 2],
            [1, 3],
            [1, 4],
            [4, 2],
            [4, 3],
            [4, 4],
        ]
        self.two_disable = [
            [0, 1],
            [2, 1],
            [3, 1],
            [5, 1],
        ]
        self.four_disable = [
            [0, 5],
            [2, 5],
            [3, 5],
            [4, 6],
            [5, 5],
        ]

        self.black = []
        for i in range(6):
            for j in range(7):
                if i == 0 or i == 2 or i == 4:
                    if 2 <= j <= 4:
                        self.black.append([i, j])

    def step(
        self,
        actions,
    ):
        # 次の観測を計算する
        # 移動の優先順位は1->2->3->4である.
        reward_list = [0 for _ in range(4)]
        for i, act_i in enumerate(actions):
            stay = False
            dummy_i = copy.deepcopy(self.agent_coor[i])
            # 既に移動先に他のエージェントがいるかどうかはとりあえず無視。
            # それが枠を飛び出すかどうかも無視
            if act_i == 1 and dummy_i not in self.corridor:
                dummy_i[0] -= 1
            elif act_i == 1 and dummy_i in self.corridor:
                reward_list[i] -= 0.001
            elif act_i == 2 and dummy_i not in self.two_disable:
                dummy_i[1] += 1
            elif act_i == 2 and dummy_i in self.two_disable:
                reward_list[i] -= 0.001
            elif act_i == 3 and dummy_i not in self.corridor:
                dummy_i[0] += 1
            elif act_i == 3 and dummy_i in self.corridor:
                reward_list[i] -= 0.001
            elif act_i == 4 and dummy_i not in self.four_disable:
                dummy_i[1] -= 1
            elif act_i == 4 and dummy_i in self.four_disable:
                reward_list[i] -= 0.001
            # 移動先に他のエージェントがいるかどうかのチェック
            for j in range(4):
                if i == j:
                    continue
                if dummy_i == self.agent_coor[j]:
                    reward_list[i] -= 0.001
                    stay = True
            # 移動先が枠内かどうかのチェック
            if dummy_i[0] < 0 or dummy_i[0] >= 6 or dummy_i[1] < 0 or dummy_i[1] >= 7:
                reward_list[i] -= 0.001
                stay = True

            # stayフラグが立っていなければ移動.
            if not stay:
                self.agent_coor[i] = dummy_i

        # 報酬を計算する
        for i, coor_i in enumerate(self.agent_coor):
            if i == 0:
                reward_list[i] += (
                    -(
                        (5 - self.agent_coor[i][0]) ** 2
                        + (6 - self.agent_coor[i][1]) ** 2
                    )
                    / 100
                )
            elif i == 1:
                reward_list[i] += (
                    -(
                        (5 - self.agent_coor[i][0]) ** 2
                        + (0 - self.agent_coor[i][1]) ** 2
                    )
                    / 100
                )
            elif i == 2:
                reward_list[i] += (
                    -(
                        (0 - self.agent_coor[i][0]) ** 2
                        + (6 - self.agent_coor[i][1]) ** 2
                    )
                    / 100
                )
            elif i == 3:
                reward_list[i] += (
                    -(
                        (0 - self.agent_coor[i][0]) ** 2
                        + (0 - self.agent_coor[i][1]) ** 2
                    )
                    / 100
                )
        reward = sum(reward_list)
        # doneかどうかの判定. time が200を超える, あるいは全エージェント目的地に着いたら終了.
        done = False
        self.time += 1
        if self.time == 50 or reward == 0:
            done = True
        return self.get_observation(), reward, done

    def reset(
        self,
    ):
        self.time = 0

        for i in range(4):
            if i == 0:
                self.agent_coor[i] = [0, 0]
            elif i == 1:
                self.agent_coor[i] = [0, 6]
            elif i == 2:
                self.agent_coor[i] = [5, 0]
            elif i == 3:
                self.agent_coor[i] = [5, 6]

    def get_observation(
        self,
    ):
        obs = copy.deepcopy(self.agent_coor)
        # obs_dummy = [[[-1 for j in range(3)] for i in range(3)] for _ in range(4)]
        # obs = [[] for _ in range(4)]
        # for agent_id in range(4):
        #     coor_agent_x = self.agent_coor[agent_id][0]
        #     coor_agent_y = self.agent_coor[agent_id][1]
        #     for i in range(3):
        #         for j in range(3):
        #             if i == 1 and j == 1:
        #                 obs_dummy[agent_id][i][j] = 0
        #             elif (
        #                 6 > coor_agent_x + (i - 1) >= 0
        #                 and 7 > coor_agent_y + (j - 1) >= 0
        #             ):
        #                 block = [
        #                     coor_agent_x + (i - 1),
        #                     coor_agent_y + (j - 1),
        #                 ]
        #                 for other_id in range(4):
        #                     if agent_id == other_id:
        #                         continue
        #                     elif (
        #                         block == self.agent_coor[other_id]
        #                     ):  # そのブロックに他のエージェントがいた場合.
        #                         obs_dummy[agent_id][i][j] = other_id + 1
        #                     elif (
        #                         block not in self.black
        #                     ):  # そのブロックにだれもいない場合.
        #                         obs_dummy[agent_id][i][j] = 0

        #     for i in range(3):
        #         for j in range(3):
        #             obs[agent_id].append(obs_dummy[agent_id][i][j])
        #     obs[agent_id] += self.agent_coor[agent_id]

        return obs
