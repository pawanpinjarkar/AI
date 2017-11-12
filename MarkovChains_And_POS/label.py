###################################
# CS B551 Fall 2016, Assignment #3
# D. Crandall
#
# There should be no need to modify this file, although you
# can if you really want. Edit pos_solver.py instead!
#
# To get started, try running:
#
#   python ./label.py bc.train bc.test.tiny
#

from pos_scorer import Score
from pos_solver import *
import sys


# Read in training or test data file
#
def read_data(fname):
    exemplars = []
    file = open(fname, 'r');

    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [(data[0::2], data[1::2]), ]

    return exemplars


####################
# Main program
#

if len(sys.argv) < 3:
    print "Usage: one of "
    print "    ./label.py part1.2 training_file"
    print "    ./label.py part1.3 training_file \"Test sentence\""
    print "    ./label.py part2.1 training_file test_file"
    sys.exit()

part = sys.argv[1]
prob= {}

if part == "part1.2":
    print "Learning model...\n\n"
    data=read_data(sys.argv[2])
    do_part12(data)

elif part == "part1.3":
    print "Learning model...\n\n"
    data = read_data(sys.argv[2])
    do_part13(data, sys.argv[3])

elif part == "part2.1":
    (train_file, test_file) = sys.argv[2:4]

    print "Learning model...\n\n"
    solver = Solver()
    train_data = read_data(train_file)
    solver.train_part2(train_data)

    print "Loading test data..."
    test_data = read_data(test_file)

    print "Testing classifiers..."
    scorer = Score()
    Algorithms = ("Simplified", "HMM VE", "HMM MAP")
    Algorithm_labels = [str(i + 1) + ". " + Algorithms[i] for i in range(0, len(Algorithms))]
    for (s, gt) in test_data:
        outputs = {"0. Ground truth": gt}

        # run all algorithms on the sentence
        for (algo, label) in zip(Algorithms, Algorithm_labels):
            outputs[label] = solver.solve(algo, s)

        posteriors = {o: solver.posterior(s, outputs[o]) for o in outputs}

        Score.print_results(s, outputs, posteriors)

        scorer.score(outputs)
        scorer.print_scores()
        print "----"

else:
    print "Invalid part specified!"
