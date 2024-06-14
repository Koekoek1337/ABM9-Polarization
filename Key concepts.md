## Agents
The agents of the system represent the inhabitants of a simplified bi-opinionated society, where one can be fully for opinion A, fully for opinion B, or anywhere in between. The agents are represented as nodes on a graph space where every edge represents an interpersonal relation. Agents form and break edges as they me

## Opinion
An abstract value in the range $[op_{a},op_{b}]$. Agent opinions fluctuate randomly over time and change based on the opinion of their neighbors. Opinions can either be set to be initially the same for all agents, or to be randomly distributed.

## Conformity
Conformity is the degree at which an agent will adjust their opinion based on that of their neighbors.  It is a float value in the range $conf = [0,1]$ where a conformity of 0 will correspond to no change in opinion from outside influence and a conformity of 1 means that the agent will take on the mean of the mean opinion of it's neighbors(50%) and it's own (50%).

For conformities between 0 and 1, the average opinion will be the weighted average of an agent with n neighbors given by the equation
$$op_{t+1} = {\overline{op}_{neigh}conf + op_t(1 - conf)}$$
## Tolerance
Value in range $[0,1]$ that affects how likely an agent is of maintaining it's connection to an agent with conflicting opinion.

## Forming and breaking relations
Over the course of the simulation, two agents with strongly differing opinions may break of their relation. The chance of breaking a relation goes up as the difference in opinion increases. The exact method for which is yet to be decided upon, which may involve a separate tolerance stat.

The formation of relations can be implemented in multiple ways, with varying degrees of complexity.

$$
p_{break} = {|op_A - op_B| \over range } (1-tol_A)
$$
#### 1. Simple
Whenever an agent breaks a relationship, it will immediately form a new one. This formation can be sampled from the network at random, based on scale free connectivity, from the friends of it's friends or a weighted choice from either.

#### 2. More complex
After an agent breaks a relationship, it will attempt to "find" a new friend by sampling from the system at every timestep until it finds someone it "clicks" with. The chance to "click" with someone is dependent on the difference in opinion between the seeker and the target as well as the conformity of the target. 
$$p_{click} = {{(1 - {|op_A - op_B| \over range}) * tol_A * tol_B}}$$

## Polarization
https://en.wikipedia.org/wiki/Diversity_index


