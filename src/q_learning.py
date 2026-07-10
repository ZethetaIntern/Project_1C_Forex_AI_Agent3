import random


class QLearningAgent:

    def __init__(self, actions):

        self.actions = actions

        self.q_table = {}

        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1

    def get_q(self, state):

        if state not in self.q_table:

            self.q_table[state] = [0.0] * len(self.actions)

        return self.q_table[state]

    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(0, len(self.actions)-1)

        q = self.get_q(state)

        return q.index(max(q))

    def learn(
        self,
        state,
        action,
        reward,
        next_state
    ):

        q = self.get_q(state)

        next_q = self.get_q(next_state)

        q[action] = q[action] + self.alpha * (

            reward +

            self.gamma * max(next_q)

            -

            q[action]

        )