"""
Trading Analytics Dashboard
============================

Interactive Streamlit dashboard for exploring:
1. Predictive model results
2. Trader behavioral clusters
3. Sentiment analysis
4. Performance metrics

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Trading Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all required data"""
    try:
        df_sentiment = pd.read_csv('fear_greed_index.csv')
        df_trades = pd.read_csv('historical_data.csv')
        
        # Convert dates
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
        df_trades['Timestamp'] = pd.to_datetime(df_trades['Timestamp'])
        df_trades['date'] = df_trades['Timestamp'].dt.date
        df_trades['date'] = pd.to_datetime(df_trades['date'])
        
        # Daily aggregations
        daily_metrics = df_trades.groupby('date').agg({
            'Closed PnL': ['sum', 'mean', 'count'],
            'Size USD': 'sum',
            'Side': lambda x: (x == 'Long').sum() / len(x) if len(x) > 0 else 0.5
        }).reset_index()
        
        daily_metrics.columns = ['date', 'daily_pnl', 'avg_pnl', 'num_trades', 
                                 'total_volume', 'long_bias']
        
        # Merge with sentiment
        df_merged = pd.merge(daily_metrics, df_sentiment, on='date', how='inner')
        
        return df_sentiment, df_trades, df_merged
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


@st.cache_resource
def load_models():
    """Load trained ML models"""
    try:
        profitability_model = joblib.load('profitability_model.pkl')
        volatility_model = joblib.load('volatility_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        sentiment_encoder = joblib.load('sentiment_encoder.pkl')
        return profitability_model, volatility_model, label_encoder, sentiment_encoder
    except:
        return None, None, None, None


@st.cache_data
def load_clusters():
    """Load clustering results"""
    try:
        clusters = pd.read_csv('trader_clusters.csv')
        return clusters
    except:
        return None


def plot_sentiment_timeline(df_merged):
    """Plot sentiment and PnL over time"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Fear & Greed Index Over Time', 'Daily PnL Over Time'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Sentiment timeline
    sentiment_colors = {
        'Extreme Fear': '#d62728',
        'Fear': '#ff7f0e',
        'Neutral': '#7f7f7f',
        'Greed': '#2ca02c',
        'Extreme Greed': '#1f77b4'
    }
    
    for sentiment in df_merged['classification'].unique():
        df_subset = df_merged[df_merged['classification'] == sentiment]
        fig.add_trace(
            go.Scatter(
                x=df_subset['date'],
                y=df_subset['value'],
                mode='markers',
                name=sentiment,
                marker=dict(color=sentiment_colors.get(sentiment, 'gray'), size=8),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # PnL timeline
    colors = ['green' if x > 0 else 'red' for x in df_merged['daily_pnl']]
    fig.add_trace(
        go.Bar(
            x=df_merged['date'],
            y=df_merged['daily_pnl'],
            name='Daily PnL',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="F&G Index Value", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)
    
    fig.update_layout(height=700, hovermode='x unified')
    
    return fig


def plot_sentiment_performance(df_merged):
    """Plot performance by sentiment category"""
    sentiment_perf = df_merged.groupby('classification').agg({
        'daily_pnl': ['mean', 'sum', 'count'],
        'num_trades': 'mean'
    }).reset_index()
    
    sentiment_perf.columns = ['sentiment', 'avg_pnl', 'total_pnl', 'days', 'avg_trades']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Daily PnL by Sentiment', 'Trading Activity by Sentiment')
    )
    
    # Average PnL
    colors = ['green' if x > 0 else 'red' for x in sentiment_perf['avg_pnl']]
    fig.add_trace(
        go.Bar(x=sentiment_perf['sentiment'], y=sentiment_perf['avg_pnl'],
               marker_color=colors, name='Avg PnL', showlegend=False),
        row=1, col=1
    )
    
    # Trading activity
    fig.add_trace(
        go.Bar(x=sentiment_perf['sentiment'], y=sentiment_perf['avg_trades'],
               marker_color='steelblue', name='Avg Trades', showlegend=False),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Sentiment", row=1, col=1)
    fig.update_xaxes(title_text="Sentiment", row=1, col=2)
    fig.update_yaxes(title_text="Average Daily PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Average # Trades", row=1, col=2)
    
    fig.update_layout(height=400)
    
    return fig


def plot_cluster_distribution(clusters):
    """Plot cluster distribution and characteristics"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cluster Distribution', 'PCA Visualization'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Cluster sizes
    cluster_counts = clusters['archetype'].value_counts()
    fig.add_trace(
        go.Bar(x=cluster_counts.index, y=cluster_counts.values,
               marker_color='lightblue', showlegend=False),
        row=1, col=1
    )
    
    # PCA scatter
    for archetype in clusters['archetype'].unique():
        cluster_data = clusters[clusters['archetype'] == archetype]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['pca1'],
                y=cluster_data['pca2'],
                mode='markers',
                name=archetype,
                marker=dict(size=10)
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Archetype", row=1, col=1)
    fig.update_xaxes(title_text="PC1", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=2)
    
    fig.update_layout(height=400)
    
    return fig


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<p class="main-header">📊 Trading Analytics Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df_sentiment, df_trades, df_merged = load_data()
        models = load_models()
        clusters = load_clusters()
    
    if df_merged is None:
        st.error("❌ Unable to load data. Please ensure CSV files are in the same directory.")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Date range filter
    min_date = df_merged['date'].min()
    max_date = df_merged['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        df_filtered = df_merged[
            (df_merged['date'] >= pd.Timestamp(date_range[0])) &
            (df_merged['date'] <= pd.Timestamp(date_range[1]))
        ]
    else:
        df_filtered = df_merged
    
    # Sentiment filter
    sentiments = st.sidebar.multiselect(
        "Filter by Sentiment",
        options=df_merged['classification'].unique(),
        default=df_merged['classification'].unique()
    )
    
    df_filtered = df_filtered[df_filtered['classification'].isin(sentiments)]
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview", "🎯 Predictions", "👥 Trader Archetypes", "📊 Deep Dive"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pnl = df_filtered['daily_pnl'].sum()
            st.metric("Total PnL", f"${total_pnl:,.2f}", 
                     delta=f"{total_pnl/len(df_filtered):.2f}/day")
        
        with col2:
            win_rate = (df_filtered['daily_pnl'] > 0).mean()
            st.metric("Win Rate", f"{win_rate:.1%}")
        
        with col3:
            total_trades = df_filtered['num_trades'].sum()
            st.metric("Total Trades", f"{total_trades:,}",
                     delta=f"{total_trades/len(df_filtered):.1f}/day")
        
        with col4:
            sharpe = (df_filtered['daily_pnl'].mean() / 
                     (df_filtered['daily_pnl'].std() + 1e-6))
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        st.markdown("---")
        
        # Timeline plots
        st.subheader("Market Sentiment & Performance Timeline")
        fig_timeline = plot_sentiment_timeline(df_filtered)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Performance by sentiment
        st.subheader("Performance Analysis by Market Sentiment")
        fig_sentiment = plot_sentiment_performance(df_filtered)
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # TAB 2: PREDICTIONS
    with tab2:
        st.header("Next-Day Predictions")
        
        if all(models):
            profitability_model, volatility_model, label_encoder, sentiment_encoder = models
            
            st.markdown("""
            Use the sliders below to input current market conditions and get predictions 
            for next-day trading outcomes.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Market Conditions")
                
                fg_value = st.slider("Fear & Greed Index", 0, 100, 50)
                sentiment_class = st.selectbox(
                    "Sentiment Classification",
                    options=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
                )
                num_trades = st.slider("Today's # Trades", 0, 100, 20)
                avg_pnl = st.slider("Avg PnL per Trade ($)", -500, 500, 50)
            
            with col2:
                st.subheader("📈 Trading Behavior")
                
                pnl_std = st.slider("PnL Std Dev ($)", 0, 1000, 200)
                total_vol = st.slider("Total Volume ($)", 0, 200000, 50000)
                avg_pos = st.slider("Avg Position Size ($)", 0, 10000, 2000)
                long_bias = st.slider("Long Bias %", 0.0, 1.0, 0.6)
            
            # Rolling metrics (using defaults)
            rolling_pnl = st.slider("7-day Avg PnL ($)", -1000, 1000, 300)
            rolling_vol = st.slider("7-day Volatility ($)", 0, 500, 150)
            rolling_trades = st.slider("7-day Avg Trades", 0, 50, 20)
            win_rate = st.slider("7-day Win Rate", 0.0, 1.0, 0.55)
            
            if st.button("🔮 Generate Prediction", type="primary"):
                try:
                    # Prepare features
                    features = pd.DataFrame([{
                        'value': fg_value,
                        'sentiment_encoded': sentiment_encoder.transform([sentiment_class])[0],
                        'num_trades': num_trades,
                        'avg_trade_pnl': avg_pnl,
                        'pnl_std': pnl_std,
                        'total_volume': total_vol,
                        'avg_position_size': avg_pos,
                        'long_bias': long_bias,
                        'rolling_pnl_7d': rolling_pnl,
                        'rolling_vol_7d': rolling_vol,
                        'rolling_trades_7d': rolling_trades,
                        'win_rate_7d': win_rate
                    }])
                    
                    # Predictions
                    profit_pred = profitability_model.predict(features)[0]
                    profit_proba = profitability_model.predict_proba(features)[0]
                    vol_pred = volatility_model.predict(features)[0]
                    
                    profit_class = label_encoder.inverse_transform([profit_pred])[0]
                    
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Next-Day Profitability")
                        st.success(f"**Predicted: {profit_class}**")
                        
                        # Probability chart
                        prob_df = pd.DataFrame({
                            'Category': label_encoder.classes_,
                            'Probability': profit_proba
                        })
                        fig_prob = px.bar(prob_df, x='Category', y='Probability',
                                        title='Confidence Distribution',
                                        color='Probability',
                                        color_continuous_scale='RdYlGn')
                        st.plotly_chart(fig_prob, use_container_width=True)
                    
                    with col2:
                        st.markdown("### Expected Volatility")
                        st.info(f"**${vol_pred:.2f}**")
                        
                        # Risk gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=vol_pred,
                            title={'text': "PnL Volatility"},
                            gauge={
                                'axis': {'range': [0, 1000]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 200], 'color': "lightgreen"},
                                    {'range': [200, 400], 'color': "yellow"},
                                    {'range': [400, 1000], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 500
                                }
                            }
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    
                    if profit_class == 'High Profit' and vol_pred < 300:
                        st.success("✅ Favorable conditions detected. Consider normal or slightly increased position sizing.")
                    elif profit_class == 'Loss' or vol_pred > 500:
                        st.warning("⚠️ Challenging conditions. Reduce position sizes and trade selectively.")
                    else:
                        st.info("ℹ️ Moderate conditions. Maintain standard risk management practices.")
                    
                except Exception as e:
                    st.error(f"Error generating prediction: {e}")
        else:
            st.warning("⚠️ Predictive models not found. Run `predictive_model.py` first to train models.")
    
    # TAB 3: TRADER ARCHETYPES
    with tab3:
        st.header("Trader Behavioral Archetypes")
        
        if clusters is not None:
            st.markdown("""
            Traders are clustered into behavioral archetypes based on their trading patterns,
            risk appetite, and performance characteristics.
            """)
            
            # Cluster visualization
            fig_clusters = plot_cluster_distribution(clusters)
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Archetype details
            st.subheader("Archetype Profiles")
            
            archetype_stats = clusters.groupby('archetype').agg({
                'avg_daily_pnl': 'mean',
                'win_rate': 'mean',
                'sharpe_ratio': 'mean',
                'avg_trades_per_day': 'mean',
                'risk_appetite': 'mean'
            }).reset_index()
            
            for _, row in archetype_stats.iterrows():
                with st.expander(f"📌 {row['archetype']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg Daily PnL", f"${row['avg_daily_pnl']:.2f}")
                        st.metric("Win Rate", f"{row['win_rate']:.1%}")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{row['sharpe_ratio']:.2f}")
                        st.metric("Trades/Day", f"{row['avg_trades_per_day']:.1f}")
                    
                    with col3:
                        st.metric("Risk Appetite", f"{row['risk_appetite']:.2f}")
        else:
            st.warning("⚠️ Clustering results not found. Run `clustering_analysis.py` first.")
    
    # TAB 4: DEEP DIVE
    with tab4:
        st.header("Deep Dive Analysis")
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        
        corr_cols = ['daily_pnl', 'num_trades', 'total_volume', 'long_bias', 'value']
        corr_matrix = df_filtered[corr_cols].corr()
        
        fig_corr = px.imshow(corr_matrix,
                            labels=dict(color="Correlation"),
                            color_continuous_scale='RdBu',
                            aspect="auto",
                            title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribution analysis
        st.subheader("PnL Distribution Analysis")
        
        fig_dist = make_subplots(
            rows=1, cols=2,
            subplot_titles=('PnL Distribution', 'PnL by Sentiment (Violin)')
        )
        
        fig_dist.add_trace(
            go.Histogram(x=df_filtered['daily_pnl'], nbinsx=50, 
                        name='PnL Distribution', showlegend=False),
            row=1, col=1
        )
        
        for sentiment in df_filtered['classification'].unique():
            df_subset = df_filtered[df_filtered['classification'] == sentiment]
            fig_dist.add_trace(
                go.Violin(y=df_subset['daily_pnl'], name=sentiment, box_visible=True),
                row=1, col=2
            )
        
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Raw data
        st.subheader("📋 Data Explorer")
        
        if st.checkbox("Show raw data"):
            st.dataframe(df_filtered, use_container_width=True)
            
            # Download button
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name=f"trading_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
