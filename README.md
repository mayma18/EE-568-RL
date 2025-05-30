# EE-568-RL

In this project, we evaluate the performance of two Reinforcement Learning with Human Feedback (**RLHF**) algorithms, Direct Preference Optimization (**DPO**) and **PPO**-based RLHF, on three OpenAI Gym environments: CartPole, Pendulum, and MountainCar.

Using preference data generated from expert and suboptimal policies, we analyze how the two algorithms perform across different preference dataset sizes and random seeds, providing a comparative view of their effectiveness and robustness. We find that PPO-based RLHF can outperform the expert when the preference dataset is large, while DPO shows instability when trained from scratch and benefits from regularization.

#### Code Structure

+ `cartpole.ipynb`: Jupyter notebook for the entire workflow of training and evaluating DPO and PPO-based RLHF on the CartPole environment.
+ `pendulum.ipynb`: Jupyter notebook for the entire workflow of training and evaluating DPO and PPO-based RLHF on the Pendulum environment.
+ `mountaincar.ipynb`: Jupyter notebook for the entire workflow of training and evaluating DPO and PPO-based RLHF on the MountainCar environment.