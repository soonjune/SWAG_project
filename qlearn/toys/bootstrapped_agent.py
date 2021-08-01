

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from qlearn.toys.model import BoostrappedDQN
from collections import Counter


class BootstrappedAgent():
    def __init__(self, args, env):
        self.action_space = env.action_space.n
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.nheads = args.nheads
        self.double_q = args.double_q
        self.rate = args.hyperparameter

        self.online_net = BoostrappedDQN(args, self.action_space)
        if args.model and os.path.isfile(args.model):
            self.online_net.load_state_dict(torch.load(args.model))
        self.online_net.train()

        self.target_net = BoostrappedDQN(args, self.action_space)
        self.update_target_net()
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)
        if args.cuda:
            self.online_net.cuda()
            self.target_net.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if args.cuda else torch.ByteTensor
        self.left_vals = []
        self.right_vals = []
        self.ucb = args.ucb
        self.state_qvals = dict()
        for i in range(args.input_dim):
            self.state_qvals[i] = [[0], [0]]

    # Acts based on single state (no batch)
    def act_single_head(self, state, k):
        self.online_net.eval()
        state = Variable(self.FloatTensor(state / 255.0))
        return self.online_net.forward_single_head(state, k).data.max(1)[1][0]

    def act(self, state):
        self.online_net.eval()
        state = Variable(self.FloatTensor(state / 255.0))
        outputs = self.online_net.forward(state)
        current = int(sum([i > 0 for i in state.data[0]]) - 1)
        
        # for evaluating uncertainty
        self.left_vals.append(list(map(lambda x: x.data[0][0].numpy(), outputs)))
        self.right_vals.append(list(map(lambda x: x.data[0][1].numpy(), outputs)))

        self.state_qvals[current] = [self.left_vals[-1], self.right_vals[-1]]

        '''if self.ucb:
            left_UCB = np.mean(self.left_vals[-1]) + self.rate * np.std(self.left_vals[-1])
            right_UCB = np.mean(self.right_vals[-1]) + self.rate * np.std(self.right_vals[-1])
            # print(left_UCB, right_UCB)
            if left_UCB > right_UCB:
                return 0
            else:
                return 1
        # logging
        # left_vals = list(map(lambda x: float(x.data[0][0].numpy()), outputs))
        # np.set_printoptions(precision=2)
        # # print("left q vals: ", np.array(left_vals))
        # # print("left mean/std", np.mean(left_vals), np.std(left_vals))
        # right_vals = list(map(lambda x: float(x.data[0][1].numpy()), outputs))
        # if np.mean(right_vals) < 1:
        #     print("right q vals: ", np.array(right_vals))
        #     print("right mean/std", np.mean(right_vals), np.std(right_vals))
        #     print("\n")
        else:'''
        actions = []
        for k in range(self.online_net.nheads):
            actions.append(int(outputs[k].data.max(1)[1][0]))
        action, _ = Counter(actions).most_common()[0]
        return action

    # Acts with an epsilon-greedy policy
    def act_e_greedy(self, state, k, epsilon=0.01):
        return random.randrange(self.action_space) if random.random() < epsilon else self.act_single_head(state, k)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def learn(self, states, actions, rewards, next_states, terminals, k):
        self.online_net.train()
        self.target_net.eval()
        states = Variable(self.FloatTensor(states))
        actions = Variable(self.LongTensor(actions))
        next_states = Variable(self.FloatTensor(next_states))
        rewards = Variable(self.FloatTensor(rewards)).view(-1, 1)
        terminals = Variable(self.FloatTensor(terminals)).view(-1, 1)

        # import pdb
        # pdb.set_trace()
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        online_outputs = self.online_net.heads[k](states)
        target_outputs = self.target_net.heads[k](next_states)
        loss = 0
        # import pdb
        # pdb.set_trace()
        state_action_values = online_outputs.gather(1, actions.view(-1, 1))

        # Compute V(s_{t+1}) for all next states.
        if self.double_q:
            next_actions = online_outputs.max(1)[1]
            next_state_values = target_outputs.gather(1, next_actions.view(-1, 1))
        else:
            next_state_values = target_outputs.max(1)[0].view(-1, 1)

        target_state_action_values = rewards + (1 - terminals) * self.discount * next_state_values.view(-1, 1)
        # Compute Huber loss
        loss += F.smooth_l1_loss(state_action_values, target_state_action_values.detach())
        # loss /= args.nheads
        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        for param in self.online_net.heads[k].parameters():
            param.grad.data.clamp_(-1, 1)

        # print parameters
        # for name, param in self.online_net.heads[k].named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        self.optimiser.step()
        return loss