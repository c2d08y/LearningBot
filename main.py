import threading
from torch.utils.tensorboard import SummaryWriter
from agent import PPOAgent
from onsite_env import OnSiteEnv
from offsite_env import OffSiteEnv
from normalization import *
from replay_buffer import *
from const import *


def main(offline_train=True):
    env = OffSiteEnv() if offline_train else OnSiteEnv()

    total_steps = 0  # 记录总步数

    args = {
        "batch_size": 50,
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
    }

    agent = PPOAgent(args)
    agent.warm_up()

    def update_model():
        agent.learn(replay_buffer, total_steps)

    def save_model():
        agent.save()

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
        total_reward = 0
        _step = total_steps
        while not done:
            a, a_log_prob = agent.predict(s)
            s_, r, done, _ = env.step(a)
            total_reward += r

            env.render("human")

            s_ = state_norm(s_).to(device)
            r = reward_scaling(r)

            replay_buffer.store(s, a, a_log_prob, r, s_, done)
            s = s_
            total_steps += 1

            # 缓存到达batch size的时候更新参数
            if len(replay_buffer) == args["batch_size"]:
                _t1 = threading.Thread(target=update_model)
                _t1.start()
                replay_buffer.clear()

            # 自动保存模型 batch_size和autosave step的最小公倍数尽量大 因为同时保存和更新比较耗时间
            if total_steps % args["autosave_step"] == 0:
                _t2 = threading.Thread(target=save_model)
                _t2.start()

        # 绘制reward曲线 代表学习效果
        if env.episode % 10 == 0:
            writer.add_scalar(f"offline_train_{env.mode}", total_reward, env.episode)

        game_result = "won" if env.win_check() == 2 else "lost"
        print(f"game {env.episode}: bot " + game_result + f", total_reward={total_reward}, step={total_steps - _step}")

        if env.quit_signal():
            break


if __name__ == '__main__':
    main()


"""
检查mask是否生效 现在似乎没有起作用
"""