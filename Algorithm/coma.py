import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Memory:
    def __init__(
        self,
        agent_num,
        action_dim,
    ):
        self.agent_num = agent_num
        self.action_dim = action_dim

        self.actions = []
        self.observations = []
        self.reward = []
        self.done = []

    def clear(self):
        self.actions = []
        self.observations = []
        self.reward = []
        self.done = []


class Actor(nn.Module):  # state_dim -> action_dim の関数
    def __init__(
        self,
        state_dim,
        action_dim,
    ):
        super(Actor, self).__init__()
        # self.fc1 = nn.Linear(state_dim, 64)
        # self.dropout1 = nn.Dropout(p=0.2)
        self.rnn = nn.GRU(
            input_size=state_dim,
            hidden_size=64,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(p=0.2)
        self.final_layer = nn.Linear(64, action_dim)

    def forward(self, x):
        x_rnn, hidden = self.rnn(x, None)
        # x_rnn = self.dropout1(x_rnn)
        x_rnn = F.relu(x_rnn)
        output = self.final_layer(x_rnn[:, -1, :])
        # output = self.dropout2(output)
        return F.softmax(output, dim=-1)


class Critic(nn.Module):  # (3, 8, 8) -> 1 の関数
    def __init__(
        self,
    ):
        super(Critic, self).__init__()

        # self.fc1 = nn.Linear(input_dim, 64)
        self.Conv2D_1 = nn.Conv2d(
            in_channels=9, out_channels=8, kernel_size=3, padding=1
        )
        self.Conv2D_2 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=5, padding=1
        )
        self.Conv2D_3 = nn.Conv2d(
            in_channels=8,
            out_channels=3,
            kernel_size=2,
            padding=1,
        )
        self.pool = nn.MaxPool2d(2, stride=2)

        self.mlp_layer = nn.Linear(90, 1)

    def forward(self, x):
        x = self.Conv2D_1(x)
        x = F.relu(x)
        # x = self.pool(x)
        x = self.Conv2D_2(x)
        x = F.relu(x)
        # x = self.pool(x)
        x = self.Conv2D_3(x)
        x = F.relu(x)
        # x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp_layer(x)
        return x


class COMA:
    def __init__(
        self,
        agent_num,
        GRU_window_size,
        state_dim,
        action_dim,
        lr_critic,
        lr_actor,
        gamma,
        target_update_steps,
        device,
    ):
        self.agent_num = agent_num
        self.window_size = GRU_window_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma

        self.target_update_steps = target_update_steps
        self.device = device

        self.memory = Memory(agent_num, action_dim)

        self.actors = [
            Actor(
                state_dim=self.state_dim,
                action_dim=action_dim,
            ).to(self.device)
            for _ in range(agent_num)
        ]
        self.critic = Critic().to(self.device)

        self.critic_target = Critic().to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actors_optimizer = [
            torch.optim.AdamW(
                self.actors[i].parameters(),
                lr=lr_actor,
            )
            for i in range(agent_num)
        ]
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=lr_critic,
        )

        self.count = 0

        self.blackblock_list = []
        for i in range(6):
            for j in range(6):
                if i == 0 or i == 2 or i == 4:
                    if 2 <= j <= 4:
                        self.blackblock_list.append([i, j])

    def get_actions(self, observations):
        for actor in self.actors:
            actor.eval()
        observations = observations.reshape(
            self.agent_num, self.window_size, self.state_dim
        )
        actions = []
        for i in range(self.agent_num):
            dist = self.actors[i](
                observations[i]
                .reshape(
                    1,
                    self.window_size,
                    -1,
                )
                .to(self.device)
            )
            action = Categorical(dist).sample()
            # action = torch.argmax(dist, dim=1)
            actions.append(action.item())
        self.memory.observations.append(observations)
        self.memory.actions.append(torch.tensor(actions).to(self.device))

        return actions

    def train(self):
        for actor in self.actors:
            actor.train()
        # self.critic_target.train()
        for agent_id in range(self.agent_num):
            # 各エージェントについて訓練する
            critic_input = self.build_critic_input(
                self.memory.observations, self.memory.actions
            )

            Q_target = self.critic_target(critic_input)  # target networkを使うs
            observations_i = torch.stack(self.memory.observations)[:, agent_id]

            agent_current_coors = observations_i[:, -1, -2:]

            pi = self.actors[agent_id](observations_i.to(self.device))
            if np.random.rand() <= 0.01:
                print(len(pi), pi[0:3], pi[-3:])
            # baselineを計算する.
            counterfactual_Q_value = None
            for act in range(self.action_dim):
                counterfactual_critic_input = torch.clone(critic_input)

                for b in range(agent_current_coors.shape[0]):
                    # {agent_id}番目のエージェントの行動を仮想的に変更
                    counterfactual_critic_input[b][
                        4 + int(self.memory.actions[b][agent_id])
                    ][int(agent_current_coors[b][0])][
                        int(agent_current_coors[b][1])
                    ] = 0
                    counterfactual_critic_input[b][4 + act][
                        int(agent_current_coors[b][0])
                    ][int(agent_current_coors[b][1])] = 1

                # 全部concatする
                if act == 0:
                    counterfactual_Q_value = self.critic(
                        counterfactual_critic_input.to(device=self.device)
                    )
                else:
                    counterfactual_Q_value = torch.cat(
                        [
                            counterfactual_Q_value,
                            self.critic(
                                counterfactual_critic_input.to(self.device),
                            ),
                        ],
                        dim=1,
                    )

            baseline = torch.sum(pi * counterfactual_Q_value, dim=1).squeeze()
            advantage = Q_target - baseline

            taken_actions_i = torch.stack(self.memory.actions)[:, agent_id]
            log_pi = torch.log(
                pi.gather(dim=1, index=taken_actions_i.reshape(-1, 1))
            ).squeeze()

            actor_loss = -torch.mean(advantage * log_pi)
            print("agent {}'s actor loss is {}".format(agent_id, actor_loss))
            self.actors_optimizer[agent_id].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actors[agent_id].parameters(),
                1.0,
            )
            self.actors_optimizer[agent_id].step()

        # ここからcriticを評価する
        self.critic.train()
        Q_target_critic = self.critic(critic_input).squeeze()
        # TD(0)
        r = torch.zeros(len(self.memory.reward)).to(self.device)

        for t in range(len(self.memory.reward)):
            if self.memory.done[t]:
                r[t] = self.memory.reward[t]
            else:
                r[t] = self.memory.reward[t] + self.gamma * Q_target_critic[t + 1]

        critic_loss = torch.mean((r - Q_target_critic) ** 2)
        print("critic loss is", critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            5.0,
        )
        self.critic_optimizer.step()

        if self.count == self.target_update_steps:
            print("We updated the target network.")
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.count = 0
        else:
            self.count += 1
        self.memory.clear()

    def build_critic_input(self, observations, actions):
        batch_size = len(observations)

        # observations = torch.stack(observations)[:, :, -1, :].reshape(batch_size, -1)
        # actionsをone hot にする
        # actions_nonhot = torch.stack(actions)
        # actions = torch.zeros(batch_size, 4 * 5)
        # for b in range(batch_size):
        #     for i in range(4):
        #         actions[b][i * 5 + actions_nonhot[b][i]] = 1
        # critic_input = torch.cat(
        #     [
        #         observations.float().to(self.device),
        #         actions.type(torch.float32).to(self.device),
        #     ],
        #     dim=-1,
        # )

        # CNN input
        agents_coor_list = []
        for b in range(batch_size):
            agents_coor_list.append(observations[b][:, -1, -2:])
        critic_input = torch.zeros(batch_size, 4 + 5, 6, 7)
        for b in range(batch_size):
            for x in range(6):
                for y in range(7):
                    critic_input[b][0][x][y] = x + y
                    critic_input[b][1][x][y] = x + y
                    critic_input[b][2][x][y] = x + y
                    critic_input[b][3][x][y] = x + y
                    if [x, y] in self.blackblock_list:
                        critic_input[b][0][x][y] = -10
                        critic_input[b][1][x][y] = -10
                        critic_input[b][2][x][y] = -10
                        critic_input[b][3][x][y] = -10
        for b in range(batch_size):
            for agent_id in range(4):
                critic_input[b][agent_id][int(agents_coor_list[b][agent_id][0])][
                    int(agents_coor_list[b][agent_id][1])
                ] = -1
                critic_input[b][4 + actions[b][agent_id]][
                    int(agents_coor_list[b][agent_id][0])
                ][int(agents_coor_list[b][agent_id][1])] = 1
        return critic_input.to(self.device)
