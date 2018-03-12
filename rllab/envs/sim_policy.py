import joblib
import cassie2d as cassie
import time


def run_policy():
    # filename = "params_exp1.pkl"
    filename = "params_exp2.pkl"

    data = joblib.load(filename)
    policy = data['policy']
    agent = policy
    env = cassie.Cassie2dEnv()

    animated = False
    speedup = 1

    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    max_path_length = 5000

    o = env.reset()
    policy.reset()
    path_length = 0

    if animated:
        env.render()

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
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
