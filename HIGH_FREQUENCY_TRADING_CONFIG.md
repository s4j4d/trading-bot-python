# High-Frequency Trading Bot Configuration

## Goal
Create an active day trading bot that:
- Makes multiple trades per day
- Capitalizes on short-term volatility
- Buys dips and sells peaks
- Operates 24/7 without human intervention

## Key Parameter Changes

### 1. GAMMA (Discount Factor) ⭐⭐⭐ MOST IMPORTANT
**Changed from:** 0.99 (long-term thinking)
**Changed to:** 0.4 (short-term trading)

**Why this helps:**
- Model focuses on immediate profit opportunities (next 5-10 steps)
- Less concerned about long-term trends
- Encourages quick buy-sell cycles
- Perfect for volatile markets

**Gamma Guide:**
- 0.99 = Long-term investor (HODL strategy)
- 0.85 = Swing trader (days to weeks)
- 0.7 = Position trader (hours to days)
- 0.4-0.5 = Day trader (minutes to hours) ✅ YOUR GOAL
- 0.1-0.3 = Scalper (seconds to minutes)

### 2. WINDOW_SIZE (Observation Window)
**Changed from:** 50 timesteps
**Changed to:** 30 timesteps

**Why this helps:**
- Shorter memory = faster reactions
- Focuses on recent price action
- Better for catching quick reversals
- Less influenced by old data

### 3. EPSILON_END (Exploration Rate)
**Changed from:** 0.05 (5% exploration)
**Changed to:** 0.15 (15% exploration)

**Why this helps:**
- Maintains more exploration during trading
- Prevents getting stuck in "never trade" mode
- Encourages trying new trading opportunities
- Balances exploitation with discovery

## Additional Recommendations

### 4. Reward Function Modifications (Optional)
Consider modifying your environment's reward function to:

```python
# Reward successful trades
if action == SELL and profit > 0:
    reward = profit * 2  # Double reward for profitable trades
    
# Penalize holding cash too long in volatile markets
if position == 0 and steps_without_position > 10:
    reward -= 0.01  # Small penalty for inactivity
    
# Reward trading activity
if trade_executed:
    reward += 0.001  # Small bonus for taking action
```

### 5. Training Data Requirements
For high-frequency trading, you need:
- ✅ High volatility periods (lots of ups and downs)
- ✅ Mix of bull and bear markets
- ✅ At least 20,000+ data points
- ✅ 5-minute or 1-hour candles (not daily)

**Your current data:**
- ❌ 34% decline (too bearish)
- ✅ High volatility (good!)
- ✅ 16,400 data points (good!)

**Recommendation:** Mix your current data with some bullish periods

### 6. Expected Behavior After Retraining

With GAMMA = 0.4, your bot should:
- ✅ Make 5-20 trades per day
- ✅ Buy when price dips 2-5%
- ✅ Sell when price rises 2-5%
- ✅ Re-enter positions after selling
- ✅ Adapt to changing market conditions

### 7. Risk Management

**Important:** Lower gamma = more aggressive trading
- Higher transaction fees (more trades)
- More slippage exposure
- Potential overtrading in low-volatility periods

**Monitor these metrics:**
- Win rate (target: >55%)
- Profit factor (target: >1.5)
- Average trade duration (target: 1-6 hours)
- Trades per day (target: 5-20)

## Training Command

After making these changes, retrain your model:

```bash
python training/trainer.py
```

## Testing the New Model

Test on different market conditions:

```bash
# Test on volatile period
python evaluate_nobitex_dqn.py --pair WIRT --days 7

# Test on longer period
python evaluate_nobitex_dqn.py --pair WIRT --days 30

# Test on different pairs
python evaluate_nobitex_dqn.py --pair ETHIRT --days 14
```

## Expected Results

**Before (GAMMA=0.99):**
- 2 trades total
- 0% buy signals
- 60% sell signals
- Holds cash forever

**After (GAMMA=0.4):**
- 50-200 trades total (for 7 days)
- 25-35% buy signals
- 25-35% sell signals
- 30-50% hold signals
- Active trading throughout period

## Troubleshooting

**If bot still doesn't trade enough:**
1. Lower gamma further (try 0.3)
2. Increase epsilon_end to 0.2
3. Reduce window_size to 20
4. Check reward function encourages trading

**If bot trades too much (overtrading):**
1. Increase gamma slightly (try 0.5)
2. Decrease epsilon_end to 0.1
3. Add minimum holding period in environment

**If bot loses money:**
1. Check training data has profitable patterns
2. Increase window_size for better context
3. Adjust reward function to penalize losses more
4. Consider gamma = 0.5-0.6 (more balanced)

## Comparison: Different Trading Styles

| Style | Gamma | Window | Trades/Day | Hold Time |
|-------|-------|--------|------------|-----------|
| HODL | 0.99 | 100 | 0-1 | Days-Weeks |
| Swing | 0.85 | 50 | 1-3 | Hours-Days |
| Day Trade | 0.4 | 30 | 5-20 | Minutes-Hours ✅ |
| Scalp | 0.2 | 10 | 50+ | Seconds-Minutes |

## Next Steps

1. ✅ Parameters updated in constants.py
2. ⏳ Retrain model with new parameters
3. ⏳ Test on evaluation data
4. ⏳ Monitor trading frequency and profitability
5. ⏳ Fine-tune based on results

Good luck with your high-frequency trading bot! 🚀
