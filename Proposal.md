# Mitigating Polarization through Cross-Group Contact:  Insights from Agent-Based Simulations  
## Abstract
Conformity is a fundamental aspect of social behaviour, characterized by the tendency  
of individuals to align their opinions with those of a group. Conformity plays a critical role in  
maintaining social cohesion; however, it can also contribute to suppression of individuality and  
dissent. We suspect that conformity pressure plays a key part in the mechanism of causing  
polarization in society.  

We plan to simulate this phenomenon with group of networked agents as a model society. A small population of ideologues, i.e., proponents of fixed opinions are perceived to influence a larger  
population of conformist followers with varying levels of conformity and tolerance to dissimilarity.  
In such a setup, polarization is bound to occur. One of the polarization mitigation strategies is to  
encourage cross-group contact (**Allport, 1954**). However, in a society with high conformity  
pressure, it can have counter-intuitive consequences due to rejection of alien opinion (**Stephan &  
Morrison, 2009**). We wish to study this relation between the population of ideologues, the  
conformity pressure and tolerance threshold, and polarization in a dual-opinionated society.  

Agents in a network are initialized and randomly assigned initial opinions and conformity levels.  
Some of the agents in the network are marked as ideologues with fixed binary opinions. The  
Shannon entropy and variance are then calculated to measure opinion polarization. At each  
iteration of simulation, the network is perturbed to force cross-group contact by randomly changing some connections between the agents of opposite groups. The conformist followers update their  
opinion by modifying, retaining or aggravating their opinion subscription based on the cumulative effect of their conformity pressure and tolerance threshold over a neighbourhood or group. The  
simulation is then run over a large number of iterations to observe a change in polarization.  

We believe that this study will shed some light on the effect of cross-group contact in a society  
with high conformity pressure vs a society with low conformity pressure, with the ultimate aim of  
establishing the efficacy of cross-group contact as a successful polarization mitigation strategy.

## Justification for ABM: 
Agent-based modelling (*ABM*) is the best approach for this study of the  
dynamics of conformity and polarization in a networked society (**Axelrod, R.1998**). ABM allows for  
the simulation of heterogeneous individual agents with unique characteristics, such as varying  
levels of conformity pressure, tolerance threshold, and response pattern to influences. This  
granularity is crucial for studying the emergence of polarization as an aggregate phenomenon  
based on complex interactions between agents. It is also possible to test different scenarios by  
experimenting with parameters like conformity pressure, tolerance levels, and network structures.  

ABM allows the incorporation of different types of agents with unique characteristics and decision- making processes (proponents, followers), which is essential for understanding how various  
agents contribute to polarization. The model can simulate interactions among agents on the  
network, capturing the effects of both local and global influences on opinion dynamics.  

Unlike other approaches, ABM can help in modelling specific network topologies, capturing the  
influence of social structures on opinion dynamics. This is particularly important for studying how  
network connectivity and clustering affect the spread of opinions and the formation of polarized  
groups. We can see a clear connection between micro-level behaviours of individual decision-  
making and macro-level outcomes at the level of societal shifts, such as the emergence or  
mitigation of polarization.  

ABM also allows agents to adapt over time, like change their opinion, which is essential for  
studying dynamic processes like opinion formation and change. This adaptability helps in  
understanding long-term trends and the potential for sustained polarization or depolarization and also in studying how opinions cluster, spread, or diverge over time. There is also a possibility of modelling spatial dimension along with temporal dimension to induce space-dependent  
interaction dynamics.
