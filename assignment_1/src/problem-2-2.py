import matplotlib.pyplot as plotter 
import timeit
import numpy as np

timer = []
steps = "5000000"

def timecalculator(n):
    for i in range(n):
        workers = str(2**i)
        time = timeit.timeit('montecarlo.compute_pi(args)',
                    setup = 'import importlib;' 
                            'montecarlo = importlib.import_module("mp-pi-montecarlo-pool");' 
                            'import argparse;'
                            'parser = argparse.ArgumentParser(description="Compute Pi using Monte Carlo simulation.");' 
                            'parser = argparse.ArgumentParser(description="Compute Pi using Monte Carlo simulation.");'
                            'parser.add_argument("--steps", default=' + steps + ', type = int);'
                            'parser.add_argument("-workers", default=' + workers +', type = int);'
                            'args = parser.parse_args();', 
                    number = 1)
        yield time

gen = timecalculator(6)
for i in range(6):
    timer.append(next(gen))

listingTime = np.array(timer)

x1 = [1, 2, 4, 8, 16, 32]
y1 = listingTime
plotter.plot(x1, y1[0]/y1, label = "measured") 

x2 = [1, 2, 4, 8, 16, 32]
y2 = x2
plotter.plot(x2, y2, label = "theoretical") 
  
plotter.xlabel('Cores') 
plotter.ylabel('Speedup') 
plotter.title('Parallel Computing: Theoretical vs. Measured') 
  
plotter.legend() 
plotter.show()