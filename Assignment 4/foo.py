import mdp
import problem_utils
m1 = mdp.makeRNProblem()
#m1.valueIteration()
m1.printValues()
m1.calculateUtility((0,0))