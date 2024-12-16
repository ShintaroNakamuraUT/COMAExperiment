import copy
import os
import pickle

import torch

from Algorithm.coma import COMA
from environment import Env

if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    print("Use", device)
    window_size_gru = 5
    agent_num = 4
    state_dim = 2
    agents = COMA(
        agent_num=agent_num,
        GRU_window_size=window_size_gru,
        state_dim=state_dim,
        action_dim=5,
        lr_critic=5e-3,
        lr_actor=1e-4,
        gamma=0.99,
        target_update_steps=20,
        device=device,
    )
    # 学習済みモデルをロード
    actors_path = ["model/actor_{}.pth".format(i) for i in range(4)]
    critic_path = "model/critic.pth"
    critic_target_path = "model/critic_target_path.pth"
    for i in range(4):
        if os.path.exists(actors_path[i]):
            print("Loaded models for actor {}.".format(i))
            agents.actors[i].load_state_dict(torch.load(actors_path[i]))
    if os.path.exists(critic_path):
        print("Loaded models for the critic.")
        agents.critic.load_state_dict(torch.load(critic_path))
    if os.path.exists(critic_target_path):
        print("Loaded models for the critic_target.")
        agents.critic.load_state_dict(torch.load(critic_target_path))

    env = Env()
    env.reset()

    n_episodes = 100000
    reward_history = []

    for episode in range(n_episodes):
        obs = env.get_observation()
        obs = torch.tensor(obs).float()

        episode_reward, trajectory = 0, [obs]
        time = 0
        # (agent_num, episode_length, state_dim)
        obs_history = obs.clone()

        for _ in range(200 + window_size_gru):
            obs_history = torch.cat([obs_history, obs], dim=1)
        while True:
            actions = agents.get_actions(
                obs_history[:, time * state_dim : (time + window_size_gru) * state_dim]
            )

            next_obs, reward, done = env.step(actions=actions)

            trajectory.append(next_obs)

            episode_reward += reward
            agents.memory.reward.append(reward)
            agents.memory.done.append(done)
            if done:
                env.reset()
                break
            obs = torch.tensor(next_obs).float()
            obs_history[
                :,
                (time + window_size_gru)
                * state_dim : (time + window_size_gru + 1)
                * state_dim,
            ] = obs

            time += 1
        reward_history.append(episode_reward)
        if episode % 10 == 0:
            # print("Let's train!!!!!!")
            agents.train()
        if episode % 50 == 0:
            print(episode)
            print(episode_reward)
        if episode % 100 == 0:
            with open("results/reward_history_{}.pkl".format(episode), "wb") as f:
                pickle.dump(reward_history, f)
            with open("results/trajectory_{}.pkl".format(episode), "wb") as f:
                pickle.dump(trajectory, f)

            # save models
            for i in range(4):
                torch.save(agents.actors[i].state_dict(), actors_path[i])
            torch.save(agents.critic.state_dict(), critic_path)
            torch.save(agents.critic_target.state_dict(), critic_target_path)
