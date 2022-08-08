from torch import  functional as F
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from utils import *


class PPOAgent(object):

    def __init__(self, args: dict):
        self.batch_size = args["batch_size"]  # batch size
        self.lr_a = args["lr_a"]  # Learning rate of actor
        self.lr_c = args["lr_c"]  # Learning rate of critic
        self.gamma = args["gamma"]  # Discount factor
        self.lamda = args["lambda"]  # GAE parameter
        self.epsilon = args["epsilon"]  # PPO clip parameter
        self.k_epochs = args["k_epochs"]  # PPO parameter
        self.entropy_coef = args["entropy_coef"]
        self.device = args["device"]  # device

        self.pai_set = {
            20: get_model("actor", "./model/non_maze.pth", 20),
            19: get_model("actor", "./model/maze.pth", 19),
            10: get_model("actor", "./model/non_maze1v1.pth", 10),
            9: get_model("actor", "./model/maze1v1.pth", 9)
        }
        self.v_set = {
            20: get_model("critic", "./model/non_maze_critic.pth", 20),
            19: get_model("critic", "./model/maze_critic.pth", 19),
            10: get_model("critic", "./model/non_maze1v1_critic.pth", 10),
            9: get_model("critic", "./model/maze1v1_critic.pth", 9)
        }
        self.pai = self.pai_set[0]
        self.v = self.v_set[0]
        self.softmax = nn.Softmax(dim=0)
        self.optimizer_actor = torch.optim.Adam(self.pai.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.v.parameters(), lr=self.lr_c, eps=1e-5)

    def learn(self, rep, step_t):
        """
        learn from experience
        :param rep:
        :param step_t:
        :return:
        """
        s, a, a_log_prob, r, s_, dw, done = rep.to_tensor()

        # calculate GAE advantage
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.v(s)
            vs_ = self.v(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs

        # advantage normalization
        adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.batch_size, False):
                dist_now = Categorical(self.softmax(self.pai(s[index])))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_log_prob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1,
                                        surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()

                # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.pai.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.v(self.softmax(s[index]))
                critic_loss = F.mse_loss(v_target[index], v_s)

                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()

                # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.v.parameters(), 0.5)
                self.optimizer_critic.step()

        self.lr_decay(step_t)

    def lr_decay(self, total_steps):
        """
        learning rate decay
        :param total_steps:
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
        sample an action from policy network
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
        change policy and value network for a new game
        :param map_size:
        :return:
        """
        self.pai = self.pai_set[map_size]
        self.v = self.v_set[map_size]

    def warm_up(self):
        """
        warm up neural networks
        :return:
        """
        t = torch.zeros([1, 12, 20, 20]).to(self.device)
        self.softmax(self.pai(t))
        self.v(t)

    def load(self, path):
        pass

    def save(self, path):
        pass
