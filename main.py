from torch.utils.tensorboard import SummaryWriter
from agent import PPOAgent
from onsite_env import OnSiteEnv
from offsite_env import OffSiteEnv
from normalization import *
from replay_buffer import *


def main(train=True):
    env = OffSiteEnv() if train else OnSiteEnv()

    total_steps = 0  # Record the total steps during the training

    device = torch.device("cuda")
    args = {
        "batch_size": None,
        "state_dim": None,
        "action_dim": 5,
        "lr_a": None,
        "lr_c": None,
        "gamma": None,
        "lambda": None,
        "epsilon": None,
        "k_epochs": None,
        "entropy_coef": None,
        "device": device
    }

    agent = PPOAgent(args)
    agent.warm_up()

    # Build a tensorboard
    writer = SummaryWriter("offline_train_logs" if train else "online_train_logs")

    while True:
        s = env.reset()
        # arguments update per game
        args["state_dim"] = env.map_size
        agent.change_network(env.map_size)

        # init normalizations
        state_norm = Normalization(shape=args["state_dim"])  # Trick 2:state normalization
        reward_scaling = RewardScaling(shape=1, gamma=args["gamma"])

        replay_buffer = ReplayBuffer(args)

        s = state_norm(s)
        reward_scaling.reset()

        done = False
        while not done:
            a, a_log_prob = agent.predict(s)
            s_, r, done, _ = env.step(a)

            s_ = state_norm(s_)
            r = reward_scaling(r)

            replay_buffer.store(s, a, a_log_prob, r, s_, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if len(replay_buffer) == args["batch_size"]:
                agent.learn(replay_buffer, total_steps)
                replay_buffer.clear()

        if env.quit_signal():
            break


if __name__ == '__main__':
    main()