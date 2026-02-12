# Fear & Greed Index Trading Analysis

A quantitative analysis examining the relationship between market sentiment (Fear & Greed Index) and trading performance metrics.

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install required dependencies:**
```bash
pip install pandas matplotlib numpy jupyter
```

### Required Data Files
Ensure the following CSV files are in the same directory as the notebook:
- `fear_greed_index.csv` - Fear & Greed Index data (2,644 records)
- `historical_data.csv` - Historical trading data (211,224 records)

### Running the Analysis

1. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Open the notebook:**
   - Navigate to `arka.ipynb` in the Jupyter interface

3. **Run all cells:**
   - Click `Cell` → `Run All` or use `Shift + Enter` to execute cells sequentially

---

## Analysis Summary

### Methodology

This analysis investigates how market sentiment influences trading outcomes by:

1. **Data Integration:** Merged Fear & Greed Index data with historical trading records based on date alignment
2. **Sentiment Categorization:** Classified market conditions into five categories (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)
3. **Performance Metrics:** Calculated key trading metrics including:
   - Daily PnL (Profit and Loss)
   - Trade frequency and volume
   - Win rates and average gains/losses
   - Long vs. short position bias
4. **Comparative Analysis:** Examined trading behavior and outcomes across different sentiment regimes

### Key Insights

#### 1. Fear Drives Higher Returns (with Higher Risk)
- **Fear days** generated the highest average daily PnL
- Increased trading activity and volume during fearful periods
- Strong long-side bias suggests traders accumulate positions during market fear
- However, higher returns came with elevated downside risk

#### 2. Extreme Greed Offers Consistency
- **Extreme Greed** periods showed the highest win rates
- Lowest average daily losses among all sentiment categories
- More stable, predictable trading environment
- Lower volatility favors strategies prioritizing consistency over magnitude

#### 3. Sentiment-Driven Trading Patterns
- Market sentiment significantly influences both trade frequency and directional bias
- Fear correlates with contrarian accumulation behavior
- Greed periods show more balanced long/short positioning

### Strategy Recommendations

#### Strategy 1: "Fearful Accumulator" (Aggressive Growth)
**When to Apply:** Fear & Greed Index signals **Fear**

**Action Plan:**
- Increase trade frequency and position sizing
- Focus on long-oriented positions to capture rebound opportunities
- Implement strict risk controls:
  - Tighter stop-losses (recommended: 1-2% per position)
  - Scaled entry approach (3-4 tranches)
  - Risk-adjusted position sizing based on volatility

**Expected Outcome:** Higher PnL potential with managed downside exposure

---

#### Strategy 2: "Consistent & Cautious" (Stable Returns)
**When to Apply:** Fear & Greed Index signals **Extreme Greed**

**Action Plan:**
- Prioritize high-probability, high win-rate setups
- Target smaller but more reliable profits
- Reduce position sizes to preserve capital during lower-volatility periods
- Focus on mean-reversion and range-bound strategies

**Expected Outcome:** Lower but more consistent returns with minimal drawdowns

---

### Important Considerations

⚠️ **Leverage Data Unavailable:** These insights do not account for leverage usage. Since leverage significantly amplifies both gains and losses, incorporating leverage metrics would be essential for validating and refining these strategies in live trading.

### Future Enhancements
- Incorporate leverage data for complete risk assessment
- Build predictive models for sentiment-based trading signals
- Backtest strategies with transaction costs and slippage
- Analyze sector-specific performance under different sentiment regimes

---

## Repository Structure
```
.
├── Charts                  #Contains all the output of charts
├── Tables                  # Contains all the Table in CSV format
├── arka.ipynb              # Main analysis notebook
├── fear_greed_index.csv    # Market sentiment data
├── historical_data.csv     # Trading performance data
└── README.md              # This file
```

## License
This project is for educational and research purposes.
