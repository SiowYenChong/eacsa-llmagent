import plotly.express as px
import pandas as pd
import streamlit as st
from datetime import datetime

class EmotionVisualizer:
    def __init__(self, session_manager):
        self.session_manager = session_manager
    
    def display_analytics_dashboard(self, session):  # Add session parameter
        timeline = session.get('emotion_timeline', [])  # Use passed session
        
        if len(timeline) < 2:
            st.warning("Continue chatting to build emotion insights!")
            return
        
        try:
            # Convert timeline data to DataFrame
            df = pd.DataFrame(timeline)
            
            # Ensure numeric conversion for sentiment values
            df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
            df['emotion_intensity'] = pd.to_numeric(df['emotion_intensity'], errors='coerce')
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create visualizations
            with st.expander("📈 Emotion Timeline", expanded=True):
                self.create_emotion_timeline(df)
                
            with st.expander("🌡️ Sentiment Heatmap"):
                self.create_heatmap(df)
                
            with st.expander("📊 Summary Statistics"):
                self.show_summary_stats(df)
                
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

    def create_emotion_timeline(self, df):
        """Create timeline using actual column names"""
        fig = px.line(
            df,
            x='timestamp',
            y='sentiment_score',  # Changed from 'intensity'
            color='dominant_emotion',  # Changed from 'emotion'
            title="Sentiment Score Timeline",
            labels={
                'sentiment_score': 'Sentiment Score',
                'timestamp': 'Time',
                'dominant_emotion': 'Dominant Emotion'
            },
            markers=True
        )
        fig.update_layout(
            xaxis_title='Conversation Timeline',
            yaxis_title='Sentiment Score (-1 to 1)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    def create_heatmap(self, df):
        """Create heatmap using correct columns"""
        heatmap_df = df.pivot_table(
            index='dominant_emotion',  # Changed from 'emotion'
            columns=pd.Grouper(key='timestamp', freq='5T'),
            values='sentiment_score',  # Changed from 'intensity'
            aggfunc='mean'
        ).fillna(0)
        
        fig = px.imshow(
            heatmap_df,
            labels=dict(x="Time Window", y="Dominant Emotion", color="Sentiment"),
            title="Sentiment Distribution Heatmap",
            aspect="auto",
            color_continuous_scale='RdYlGn',
            range_color=(-1, 1)  # Assuming sentiment scores range from -1 to 1
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_summary_stats(self, df):
        """Update statistics with correct column names"""
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
            st.dataframe(
                df.groupby('dominant_emotion')['sentiment_score']
                .agg(['mean', 'max', 'min', 'count'])
                .rename(columns={
                    'mean': 'Average',
                    'max': 'Maximum',
                    'min': 'Minimum',
                    'count': 'Count'
                }),
                use_container_width=True
            )
    
    def display_emotion_flow(self):
        timeline = self.session_manager.current_session.get('emotion_timeline', [])
        df = pd.DataFrame(timeline)
        
        fig = px.line(df, x='timestamp', y='intensity', color='dominant_emotion',
                     title="Emotional Intensity Flow", markers=True)
        fig.update_layout(
            hovermode="x unified",
            yaxis_range=[-1,1],
            annotations=self._get_significant_events()
        )
        st.plotly_chart(fig)

    def _get_significant_events(self):
        events = []
        for i, entry in enumerate(self.session_manager.current_session['emotion_timeline']):
            if abs(entry['intensity']) > 0.8:
                events.append(dict(
                    x=entry['timestamp'],
                    y=entry['intensity'],
                    text=f"⚠️ {entry['dominant_emotion'].upper()}",
                    showarrow=True
                ))
        return events
    
    # Add new explanation visualization
    def display_explanations(self, explanation: dict):
        with st.expander("🧠 AI Decision Breakdown"):
            st.write("### Emotion Attribution Analysis")
            df = pd.DataFrame({
                'Token': explanation['tokens'],
                'Relevance Score': explanation['attributions']
            })
            fig = px.bar(df, x='Token', y='Relevance Score', 
                        color='Relevance Score',
                        color_continuous_scale='RdBu')
            st.plotly_chart(fig)
            
            st.write("**Key Influences:**")
            st.markdown("""
            - Words with positive scores increased emotion detection confidence
            - Negative scores indicate reducing influence on the final decision
            """)