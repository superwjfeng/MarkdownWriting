
# Definition

## *Benchmarking/Criteria of Cogntion*

### Turing Test

### Chinese Room Argument

### The Uncanny Valley

# Basic Concepts in Cognitive Science

## *Prerequisites*

## *Tools and Methods*

# Neurons and the Brain

## *Human Brain*

### The Human Neocortex

## *Modeling and Simulating the Brain*

### Whole-Brain Modeling

# Deep Neural Networks

# Reinforcement learning（详见Reinforcement Learning)

## Learning in Psychology

* Classic Conditioning 经典条件反射
  * Unconditioned stimulus: stimulus that produces reaction without any pretraining
  * Conditoined reaction: present unconditioned stimulus together with conditioned stimulus, conditoned stimus will later lead to conditioned reaciton without unconditioned simulus
* Operant Conditioning 操作性条件反射：Use certain stimulus as response to certain behaviour

## MRP $\langle\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma\rangle$

$p(s',r|s,a)=\mathbb{P}[S_t=s',R_t=r|S{t-1}=s,A_{t-1}=a]$

* Return is the cumulative reward taking discout into account $G=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots=\sum\limits_{k=t+1}^{T}{\gamma^{k-t-1}R_k}=R_{t+1}+\gamma G_{t+1}$
* Policy $\pi$ determines agengts how to select actions accoding to state
  * Deterministic policy 确定性策略 $a=\pi(s)$
  * Stochastic policy 随机策略 $\pi(a|s)=\mathbb{P}[A_t=a|S_t=s]$

### Value function

* State-value function is the expected return when a specific policy is followed after visiting a particular state: $v_{\pi}(s)\doteq\mathbb{E}_{\pi}[G_t|S_t=s]$
* Action-value function is the expected return when a specific piolicy is followed after choosing an action in a particular state: $q_{\pi}(s,a)\doteq\mathbb{E}[G_t|S_t=s,A_t=a]$

### Bellman Equations

* Bellman equation declares the recursive relationship between consecutive states: $v_{\pi}(s)\doteq\sum\limits_{a}{\pi(a|s)\sum\limits_{s',r}{p[s',r|s,a](r+\gamma v_{\pi}(s'))}}$
* Bellman equation for action-value funciton: $q_{\pi}(s,a)\doteq\sum\limits_{s'.r}{p(s',r|s,a)\left[r+\gamma\sum\limits_{a'}{\pi(a'|s')q_{\pi}(s',a')}\right]}$

### Bellman Optimality Equations

$v_*(s)=\sum\limits_{a}{\pi_*(a|s)\sum\limits_{s',r}{p[s',r|s,a](r+\gamma v_*(s'))}}$

### Generalized Policy Iteration GPI

* Value function depends on which policy used, and vice versa policy depends on what value function returns, there for we need to iteratively apply policy evaluation and policy improvement
* Policy evaluation is called prediction(classical conditioning)
* Policy evaluation & improvement is called control(operant conditioning)

## Dynamic Programming DP

### Policy Iteration

* Policy Evaluation
  * Propagating value between consecutive states by iteratively exploiting the recursive relationship that is formulated by the Bellman Equation is denoted as **Bootstrapping**
  * For a known MDP, we can simply evaluate the value function for an arbitrary policy by iteratively sweeping over ***ALL States***
  * $v_{k+1}(s)\doteq\sum\limits_{a}{\pi(a|s)\sum\limits_{s',r}{p[s',r|s,a][r+\gamma v_k(s')]}}$
* Policy Improvement: Improve policy $\pi$ by greedily acting on the value function $v$: $\pi'(s)\doteq\argmax\limits_{a}\sum\limits_{s',r}{p[s',r|s,a][r+\gamma v_{\pi}(s')]}$

### Value Iteration

* A variation of policy iteration taht is not using exhaustive evaluation step but a single sweep is called value iteration

## Monte Carlo Method MC

### MC Prediciton

Return is calculated for all states in each sampled trajectory. For the evaluation of value funcitons, experienced returns are averaged.

### MC Control

* With Exploring Starts
* Without Exploring Starts

## Temporal Difference TD

TD is a mixsture of DP and MC that samples and bootstraps. TD does not need models compared with DP; TD does not need to calculate all states in every sampled trajectory.

### TD Prediction: 1-Step TD

In TD prediction, the Bellman equation is employed by iteratively updating value after every time step $V(S_t)\leftarrow V(S_t)+\alpha[\underbrace{\overbrace{R_{t+1}+\gamma V(S_{t+1})}^{TD\ target}-V(S_t)}_{TD\ error}]$

### TD Control

* Target policy and Behaviour policy
  * Target policy is the policy that the agent estimates its value function according to expected return
  * Behaviour policy is the policy that the agent actually behave according to it to sample actions within the interaction with the environment
* SARSA On-Policy
  * On-Policy means the behaviour policy ($\epsilon$-greedy) is same as target policy ($\epsilon$-greedy)
  * $Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[\underbrace{R_{t+1}+\gamma Q(S_{t+1},A_{t+1})}_{on-policy\ target}-Q(S_t,A_t)]$
* Q-Learning Off-Policy
  * Off-Policy means the behaviour policy (greedy) differs from target policy ($\epsilon$-greedy)
  * $Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[\underbrace{R_{t+1}+\gamma\max\limits_{a}{Q(S_{t+1},a)}}_{off-policy\ target}-Q(S_t,A_t)]$

### DQN