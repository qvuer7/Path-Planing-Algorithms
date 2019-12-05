
- Path Planning
- Project reilizing two path planning and obstacle avoidance algoruthms in 3-D space using python.
- RRT24 contains Random Rapidly Exploring tree algorithm realization code. 
- PRM24 contains Probabilistic Road Map algorithm realizing code. 
- RRT24_test contains code for RRT algorithm performance(computational time, trajectory length with and without smoothness) testing, for clear computational time tests any plots removed from this code, therefore run same design scenarios for RRT24 and RRT24_test, to recieve trajectory plot from RRT24 and path planning performance from RRT24_test

Run command: <pip install -r requirement.txt> to install all requirements 

RRT takes all input from user, therefore running it from terminal makes all codes relevent fro each specific user input 
PRM do not takes any user input atm, but you can change htem inside code, such as: 
  - iterations - for number of iterations 
  - snode = node(Xstart, Ystart, ZStart)
  - gnode = node(Xgoal,Ygoal, Zgoal)
  - k_min - by default equal to number of iterations + num(gnode) + num(snode), but to decrease computational time ypu can play    with this value 
  - obstacle coordinates = [[Xobstacle,Yobstacle,Zobstacle], ....[Xobstacle(n), Yobstalce(n), Zobstacle(n)]], you can append as much obstacles as you want to. 

Further imporments will be provided.



Contact: telegream : @quitfromtheloop
