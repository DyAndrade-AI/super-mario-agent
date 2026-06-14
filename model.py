"""
Legacy import surface for the current Q-learning model.

The active architecture lives in:
- models/dueling_dqn.py
- agents/dqn_agent.py
"""

try:
    from models.dueling_dqn import DuelingDQNNetwork, create_dueling_dqn
    print("Modelo Dueling DQN cargado desde models/dueling_dqn.py")
except ImportError:
    print("Usar: from models.dueling_dqn import create_dueling_dqn")
