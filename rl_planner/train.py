"""Скрипт обучения DQN-агента для выбора нарративных вех."""
import json
from rl_planner.env import StoryEnv
from rl_planner.dqn_agent import DQNAgent
from pathlib import Path


def train(resume=False):
    """
    Параметры: resume (bool): если True, загружает ранее сохранённую модель и продолжает обучение.
    Сохраняет веса модели в 'dqn_model.pt' и порядок флагов в 'flags_order.json'.
    """
    project_root = Path(__file__).resolve().parent.parent
    scenario_path = str(project_root / "experiments" / "scenarios" / "mayor_support.json")
    env = StoryEnv(scenario_path, training=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    if resume:
        agent.load("dqn_model.pt")
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        agent.epsilon = 1

    episodes = 100
    batch_size = 32
    target_update_freq = 10

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            available = env._get_available()
            valid_actions = [env.beacon_to_idx[b["id"]] for b in available]
            if not valid_actions:
                total_reward = -5.0
                done = True
                break
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step(batch_size)
            state = next_state
            total_reward += reward
        if ep % target_update_freq == 0:
            agent.update_target()
        print(f"Episode {ep}, total reward: {total_reward:.2f}, epsilon: {agent.epsilon:.3f}")

    agent.save("dqn_model.pt")
    print("Модель сохранена.")

    flags_order = env._get_all_flags()
    with open("flags_order.json", "w") as f:
        json.dump(flags_order, f)


if __name__ == "__main__":
    train(resume=False)
