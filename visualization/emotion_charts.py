import plotly.express as px
import pandas as pd
import streamlit as st
from datetime import datetime
import json

class EmotionVisualizer:
    def __init__(self, session_manager):
        self.session_manager = session_manager
    
    def display_analytics_dashboard(self, session):
        timeline = session.get('emotion_timeline', [])
        
        if len(timeline) < 2:
            st.warning("Continue chatting to build emotion insights!")
            return
        
        try:
            df = pd.DataFrame(timeline)
            session_id = session['id']  # Get session ID for unique keys
            
            # Data preprocessing
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            df['intensity_trend'] = pd.to_numeric(df['intensity_trend'], errors='coerce')

            # Visualizations with unique keys
            with st.expander("📈 Emotion Timeline", expanded=True):
                self.create_emotion_timeline(df, session_id)
                
            with st.expander("🌡️ Sentiment Heatmap"):
                self.create_heatmap(df, session_id)
                
            with st.expander("📊 Summary Statistics"):
                self.show_summary_stats(df, session_id)
                
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

    def create_emotion_timeline(self, df, session_id):
        """Create timeline with unique session-based key"""
        fig = px.line(
            df,
            x='timestamp',
            y='sentiment_score',
            color='dominant_emotion',
            title="Sentiment Score Timeline",
            labels={
                'sentiment_score': 'Sentiment Score',
                'timestamp': 'Time',
                'dominant_emotion': 'Dominant Emotion'
            },
            markers=True
        )
        st.plotly_chart(
            fig, 
            use_container_width=True,
            key=f"timeline_{session_id}"  # Unique key per session
        )

    def create_heatmap(self, df, session_id):
        """Heatmap with session-specific key"""
        heatmap_df = df.pivot_table(
            index='dominant_emotion',
            columns=pd.Grouper(key='timestamp', freq='15T'),
            values='sentiment_score',
            aggfunc='mean'
        ).fillna(0)
        
        fig = px.imshow(
            heatmap_df,
            labels=dict(x="Time Window", y="Dominant Emotion", color="Sentiment"),
            title="Sentiment Distribution Heatmap",
            aspect="auto",
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(
            fig, 
            use_container_width=True,
            key=f"heatmap_{session_id}"  # Unique session-based key
        )

    def show_summary_stats(self, df, session_id):
        """Statistics section with unique ID"""
        if 'dominant_emotion' not in df.columns or 'sentiment_score' not in df.columns or df.empty:
            st.warning("Insufficient data to generate summary statistics.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Interactions", len(df))
            st.write("**Most Frequent Emotion**")
            st.write(df['dominant_emotion'].mode()[0])

        with col2:
            avg_sentiment = df['sentiment_score'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            st.write("**Strongest Sentiment**")
            max_idx = df['sentiment_score'].abs().idxmax()
            st.write(f"{df.loc[max_idx]['dominant_emotion']} ({df.loc[max_idx]['sentiment_score']:.2f})")

        with col3:
            st.write("**Sentiment Distribution**")
            grouped_df = df.groupby('dominant_emotion')['sentiment_score'].agg(['mean', 'max', 'min', 'count'])
            if grouped_df.empty:
                st.warning("No sentiment data available.")
            else:
                st.dataframe(
                    grouped_df.rename(columns={
                        'mean': 'Average',
                        'max': 'Maximum',
                        'min': 'Minimum',
                        'count': 'Count'
                    }),
                    use_container_width=True
                )

    def display_explanations(self, explanation: dict):
        """Explanation visualization with content-based key"""
        with st.expander("🧠 AI Decision Breakdown"):
            st.write("### Emotion Attribution Analysis")
            df = pd.DataFrame({
                'Token': explanation['tokens'],
                'Relevance Score': explanation['attributions']
            })
            
            # Generate unique key from explanation content
            explanation_hash = hash(json.dumps(explanation, sort_keys=True))
            fig = px.bar(df, x='Token', y='Relevance Score', 
                        color='Relevance Score',
                        color_continuous_scale='RdBu')
            
            st.plotly_chart(
                fig,
                key=f"explanation_{explanation_hash}"  # Unique content-based key
            )
            
            st.write("**Key Influences:**")
            st.markdown("""
            - Words with positive scores increased emotion detection confidence
            - Negative scores indicate reducing influence on the final decision
            """)