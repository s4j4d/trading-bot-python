"""
Utility functions for training the cryptocurrency trading bot.
"""


def linear_decay(step, total_done_steps, start_val, end_val):
    """
    Linearly decays a value from start_val to end_val over a specified number of steps.
    
    This function is commonly used for epsilon decay in reinforcement learning,
    where exploration probability decreases linearly over time to shift from
    exploration to exploitation.
    
    Args:
        step (int): Current step number (0-based)
        total_done_steps (int): Total number of steps over which to perform the decay
        start_val (float): Initial value at step 0
        end_val (float): Final value at step total_done_steps
        
    Returns:
        float: Linearly interpolated value between start_val and end_val
               - At step 0: returns start_val
               - At step total_done_steps: returns end_val
               - At intermediate steps: returns proportionally interpolated value
               - For steps > total_done_steps: returns end_val (clamped)
    
    Example:
        >>> # Decay epsilon from 1.0 to 0.1 over 1000 steps
        >>> epsilon = linear_decay(500, 1000, 1.0, 0.1)  # Returns 0.55
        >>> epsilon = linear_decay(1000, 1000, 1.0, 0.1)  # Returns 0.1
    """
    # Calculate the fraction of decay completed (0.0 to 1.0)
    # min() ensures we don't exceed 1.0 for steps beyond total_done_steps
    fraction = min(1.0, step / total_done_steps)
    
    # Linear interpolation: start + fraction * (end - start)
    return start_val + fraction * (end_val - start_val)