# Import model functionality
from model import build_q_network

# Import training functionality
from training.trainer import train_trading_bot

# --- Main Training Loop ---
if __name__ == "__main__":
    train_trading_bot()