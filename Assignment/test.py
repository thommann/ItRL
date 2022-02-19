import numpy as np
from matplotlib import pyplot

from Assignment.Network import depickle, epsilon_greedy_policy
from Assignment.playground import Chess, moving_average

network = depickle()
def test():
    nr_steps = []
    for i in range(2000):
        test_chess = Chess()
        count = 0
        if i % 100 == 0:
            print(i)
        while True:
            count +=1
            test_Qvalues, _ = network.forward(test_chess.state)
            test_Qvalues -= (1 - test_chess.get_valid_actions()) * 100000
            a = epsilon_greedy_policy(np.array([test_Qvalues]), 0).T
            test_chess.do_action(a)
            if test_chess.done:
                nr_steps.append(count)
                break
            if count > 1000:
                nr_steps.append(count)
            test_chess.move_b()
    pyplot.plot(moving_average(nr_steps, 200))
    pyplot.show()
test()