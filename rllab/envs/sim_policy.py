import joblib
import cassie2d as cassie
import time
from rllab.envs.normalized_env import normalize


def run_policy():
    filename = "params.pkl"

    data = joblib.load(filename)
    policy = data['policy']
    agent = policy
    env = normalize(cassie.Cassie2dEnv())

    animated = False
    speedup = 1

    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    max_path_length = 10000

    o = env.reset()
    policy.reset()
    path_length = 0

    if animated:
        env.render()

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a, n=1)       # n is the number of simulation steps to forward
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)


if __name__ == "__main__":
    run_policy()
    print("here")
