import numpy as np


class Experience:
	def __init__(self, state_before, state_after, action, reward):
		self.state_before = state_before
		self.state_after = state_after
		self.action = action
		self.reward = reward
