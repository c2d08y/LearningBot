from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from utils import *


class PPOAgent(object):

    def __init__(self, args: dict):
        self.batch_size = args["batch_size"]            # batch size
        self.lr_a = args["lr_a"]                        # 策略网络学习率
        self.lr_c = args["lr_c"]                        # 价值网络学习率
        self.gamma = args["gamma"]                      # 折扣因子
        self.lamda = args["lambda"]                     # GAE λ
        self.epsilon = args["epsilon"]                  # PPO ε
        self.k_epochs = args["k_epochs"]                # PPO 训练轮数
        self.entropy_coef = args["entropy_coef"]
        self.device = args["device"]                    # 运行设备

        # 神经网络
        self.pai_set = {
            20: get_model("actor", "./model/non_maze.pth", 20).to(self.device),
            19: get_model("actor", "./model/maze.pth", 19).to(self.device),
            10: get_model("actor", "./model/non_maze1v1.pth", 10).to(self.device),
            9: get_model("actor", "./model/maze1v1.pth", 9).to(self.device)
        }
        self.v_set = {
            20: get_model("critic", "./model/non_maze_critic.pth", 20).to(self.device),
            19: get_model("critic", "./model/maze_critic.pth", 19).to(self.device),
            10: get_model("critic", "./model/non_maze1v1_critic.pth", 10).to(self.device),
            9: get_model("critic", "./model/maze1v1_critic.pth", 9).to(self.device)
        }
        self.pai = self.pai_set[20]
        self.v = self.v_set[20]
        self.optimizer_actor = torch.optim.Adam(self.pai.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.v.parameters(), lr=self.lr_c, eps=1e-5)

        # others
        self.mse_loss_fn = nn.MSELoss()

    def learn(self, rep, step_t):
        """
        learn from experience
        :param rep:
        :param step_t:
        :return:
        """
        s, a, a_log_prob, r, s_, dw, done = rep.to_tensor()

        # 利用GAE计算优势函数
        adv = []
        gae = 0
        with torch.no_grad():  # 不需要梯度
            vs = self.v(s)
            vs_ = self.v(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs

        # 优势归一化
        adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # 参数更新k轮
        for _ in range(self.k_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.batch_size, False):
                dist_now = Categorical(self.pai(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)                       # shape(batch_size x 1)
                a_log_prob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(batch_size x 1)

                # https://www.luogu.com.cn/paste/9vwi6ls0
                # 计算策略梯度
                ratios = torch.exp(a_log_prob_now - a_log_prob[index])              # shape(batch_size x 1)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1,
                                        surr2) - self.entropy_coef * dist_entropy  # shape(batch_size x 1)
                # 更新策略网络
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.pai.parameters(), 0.5)
                self.optimizer_actor.step()

                # 价值网络梯度
                v_s = self.v(s[index])
                critic_loss = self.mse_loss_fn(v_target[index], v_s)
                # 更新价值网络
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.v.parameters(), 0.5)
                self.optimizer_critic.step()

        self.lr_decay(step_t)

    def lr_decay(self, total_steps):
        """
        学习率衰减
        :param total_steps: 已训练步数
        :return:
        """
        decay_rate = 0.1
        upt = 1 / (1 + decay_rate * total_steps)
        lr_a_now = self.lr_a * upt
        lr_c_now = self.lr_c * upt
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def predict(self, observation):
        """
        从策略网络采样动作
        :param observation: s_t
        :return: 2 tensors: action, ln(p(a_t|s_t))
        """
        with torch.no_grad():
            action_p = Categorical(self.pai(observation))
            action = action_p.sample()
            a_log_prob = action_p.log_prob(action)
        return action, a_log_prob

    def change_network(self, map_size):
        """
        当模式更换时 更换神经网络
        :param map_size:
        :return:
        """
        self.pai = self.pai_set[map_size]
        self.v = self.v_set[map_size]

    def warm_up(self):
        """
        预热 因为神经网络第一次跑会比较慢
        :return:
        """
        t = torch.zeros([1, 12, 20, 20]).to(self.device)
        self.pai(t)
        self.v(t)

    def save(self):
        # 保存策略网络
        torch.save(self.pai_set[20], "./model/non_maze.pth")
        torch.save(self.pai_set[10], "./model/non_maze1v1.pth")
        torch.save(self.pai_set[19], "./model/maze.pth")
        torch.save(self.pai_set[9], "./model/maze1v1.pth")

        # 保存价值网络
        torch.save(self.v_set[20], "./model/non_maze_critic.pth")
        torch.save(self.v_set[10], "./model/non_maze1v1_critic.pth")
        torch.save(self.v_set[19], "./model/maze_critic.pth")
        torch.save(self.v_set[9], "./model/maze1v1_critic.pth")
