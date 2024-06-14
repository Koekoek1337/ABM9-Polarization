## Opinion
(See [[Agent Properties#Opinion (float)|opinion]])
#### Fluctuation
TODO: Define function for fluctuation of opinion.
(Sample from normal distribution?)

#### Conformation
An agent's opinion at time $t + 1$ depends on the mean opinion of it's neighbors at time = t and it's own [[Agent Properties#Conformity (float)|conformity]] and is defined as: 
$$op_{t+1} = {\overline{op}_{neigh}conf + op_t(1 - conf)}$$

## Bonding
#### Breaking bonds
An agent may choose to break a relation with another based on a difference in opinion depending on their individual [[Agent Properties#Tolerance (float)|tolerance]]. The probability of breaking a bond is given by
$$
p_{break} = {|op_A - op_B| \over range } (1-tol_A)
$$

#### Making bonds
When an agent's degree is lower than its [[Agent Properties#Degree (int)|target degree]], it will attempt to bond to other agents in the system.
TODO: Probability of targeting an agent for bond formation; random sampling from system for now.
The probability for two agents depend on whether the initiator and target agents "click", which is determined by the difference in opinion between the two agents and both their tolerances. It is given as
$$p_{click} = {{(1 - {|op_A - op_B| \over range}) * tol_A * tol_B}}$$
#behavior #agents