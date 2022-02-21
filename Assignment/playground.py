from Assignment.test import test
from Assignment.train import train

print("Train:")
train("q")
print()
print("Test:")
test(filename="q-256.pcl")
