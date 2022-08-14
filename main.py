from torch.utils.tensorboard import SummaryWriter
from agent import PPOAgent
from onsite_env import OnSiteEnv
from offsite_env import OffSiteEnv
from normalization import *
from replay_buffer import *


def main(offline_train=True):
    env = OffSiteEnv() if offline_train else OnSiteEnv()

    total_steps = 0  # 记录总步数

    device = torch.device("cuda")
    args = {
        "batch_size": 100,
        "state_dim": None,
        "action_dim": 5,
        "lr_a": 0.01,
        "lr_c": 0.01,
        "gamma": 0.99,
        "lambda": 0.95,
        "epsilon": 0.2,
        "k_epochs": 10,
        "entropy_coef": 0.01,
        "autosave_step": 107,
        "device": device
    }

    agent = PPOAgent(args)
    agent.warm_up()

    # 绘图器
    writer = SummaryWriter("offline_train_logs" if offline_train else "online_train_logs")

    while True:
        s = env.reset()
        # 更新地图大小并更换神经网络
        args["state_dim"] = env.map_size
        agent.change_network(env.map_size)

        # 初始化一些用于归一化的类
        state_norm = Normalization(shape=args["state_dim"])  # Trick 2:state normalization
        reward_scaling = RewardScaling(shape=1, gamma=args["gamma"])

        replay_buffer = ReplayBuffer(args)

        s = state_norm(s).to(device)
        reward_scaling.reset()

        done = False
        while not done:
            a, a_log_prob = agent.predict(s)
            s_, r, done, _ = env.step(a)

            env.render("human")

            s_ = state_norm(s_).to(device)
            r = reward_scaling(r)

            replay_buffer.store(s, a, a_log_prob, r, s_, done)
            s = s_
            total_steps += 1

            # 缓存到达batch size的时候更新参数
            if len(replay_buffer) == args["batch_size"]:
                agent.learn(replay_buffer, total_steps)
                replay_buffer.clear()

            # 自动保存模型 batch_size和autosave step的最小公倍数尽量大 因为同时保存和更新比较耗时间
            if total_steps % args["autosave_step"] == 0:
                agent.save()

        if env.quit_signal():
            break


if __name__ == '__main__':
    main()

"""
Traceback (most recent call last):
  File "D:/MyFiles/LearningBot/main.py", line 79, in <module>
    main()
  File "D:/MyFiles/LearningBot/main.py", line 54, in main
    s_, r, done, _ = env.step(a)
  File "D:\MyFiles\LearningBot\offsite_env.py", line 85, in step
    self.execute_actions(action[0].long())
  File "D:\MyFiles\LearningBot\offsite_env.py", line 156, in execute_actions
    self.internal_bots[cur_color].bot_move()
  File "D:\MyFiles\LearningBot\bot_div\game.py", line 365, in bot_move
    self.flush_movements()
  File "D:\MyFiles\LearningBot\bot_div\game.py", line 341, in flush_movements
    act = [self.cur_x, self.cur_y, self.cur_x + directions[cur_movement][0],
KeyError: 'Z'
"""