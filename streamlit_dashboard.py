"""
Interactive Streamlit Dashboard for Multilingual App Reviews Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Configure Streamlit page
st.set_page_config(
    page_title="Multilingual App Reviews Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Fix search bars and input styling */
    .stSelectbox > div > div {
        background-color: white !important;
        color: black !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox > div > div > div {
        color: black !important;
        font-weight: 500 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: white !important;
        color: black !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        font-size: 16px !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: white !important;
        color: black !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        font-size: 16px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    try:
        df = pd.read_csv('multilingual_mobile_app_reviews_2025.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'multilingual_mobile_app_reviews_2025.csv' is in the project directory.")
        return None

@st.cache_data
def perform_sentiment_analysis(df):
    """Perform sentiment analysis and cache results."""
    def analyze_sentiment(text):
        if pd.isna(text) or not isinstance(text, str):
            return 'neutral'
            
        text_lower = text.lower()
        
        positive_words = ['amazing', 'great', 'excellent', 'love', 'best', 'perfect', 'fantastic', 'awesome', 'outstanding', 'wonderful', 'good']
        negative_words = ['terrible', 'awful', 'worst', 'hate', 'bad', 'horrible', 'useless', 'disappointing', 'buggy', 'poor', 'slow']
        
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    df_copy = df.copy()
    df_copy['sentiment'] = df_copy['review_text'].apply(analyze_sentiment)
    return df_copy

def train_rating_prediction_models(df):
    """Train multiple rating prediction models."""
    # Prepare data
    df_clean = df.dropna(subset=['review_text', 'rating'])
    
    # Text features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text = vectorizer.fit_transform(df_clean['review_text'])
    
    # Numerical features
    X_numerical = df_clean[['num_helpful_votes', 'user_age']].fillna(0)
    
    # Target
    y = df_clean['rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0)
    }
    
    # Train and evaluate models
    model_results = {}
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_test
            }
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
    
    return model_results, vectorizer

def train_sentiment_classification_models(df):
    """Train multiple sentiment classification models."""
    df_clean = df.dropna(subset=['review_text', 'sentiment'])
    
    # Text features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df_clean['review_text'])
    y = df_clean['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Train and evaluate models
    model_results = {}
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'actual': y_test
            }
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
    
    return model_results, vectorizer

def create_rating_prediction_viz(model_results, selected_model):
    """Create visualizations for rating prediction models."""
    if selected_model not in model_results:
        st.error(f"Model {selected_model} not found!")
        return
    
    result = model_results[selected_model]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs Predicted scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result['actual'],
            y=result['predictions'],
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(result['actual'].min(), result['predictions'].min())
        max_val = max(result['actual'].max(), result['predictions'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{selected_model} - Actual vs Predicted',
            xaxis_title='Actual Rating',
            yaxis_title='Predicted Rating',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residuals plot
        residuals = result['actual'] - result['predictions']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result['predictions'],
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='green', opacity=0.6)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title=f'{selected_model} - Residuals Plot',
            xaxis_title='Predicted Rating',
            yaxis_title='Residuals',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RMSE", f"{result['rmse']:.3f}")
    with col2:
        st.metric("MAE", f"{result['mae']:.3f}")
    with col3:
        st.metric("R² Score", f"{result['r2']:.3f}")

def create_sentiment_classification_viz(model_results, selected_model):
    """Create visualizations for sentiment classification models."""
    if selected_model not in model_results:
        st.error(f"Model {selected_model} not found!")
        return
    
    result = model_results[selected_model]
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(result['actual'], result['predictions'])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['negative', 'neutral', 'positive'],
        y=['negative', 'neutral', 'positive'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title=f'{selected_model} - Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Accuracy metric
    st.metric("Accuracy", f"{result['accuracy']:.3f}")

def create_eda_visualizations(df):
    """Create EDA visualizations."""
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        fig = px.histogram(df, x='rating', nbins=20, title='Rating Distribution')
        fig.add_vline(x=df['rating'].mean(), line_dash="dash", 
                     annotation_text=f"Mean: {df['rating'].mean():.2f}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Language distribution
        lang_counts = df['review_language'].value_counts().head(10)
        fig = px.bar(x=lang_counts.index, y=lang_counts.values, 
                    title='Top 10 Languages')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Country distribution
        country_counts = df['user_country'].value_counts().head(10)
        fig = px.bar(x=country_counts.index, y=country_counts.values,
                    title='Top 10 Countries')
        st.plotly_chart(fig, use_container_width=True)
        
        # App category distribution
        fig = px.pie(df, names='app_category', title='App Categories')
        st.plotly_chart(fig, use_container_width=True)

def create_time_series_viz(df):
    """Create time series visualizations."""
    df['review_date'] = pd.to_datetime(df['review_date'])
    
    # Daily review volume
    daily_counts = df.groupby(df['review_date'].dt.date).size()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_counts.index,
        y=daily_counts.values,
        mode='lines+markers',
        name='Daily Reviews'
    ))
    
    # Add 7-day moving average
    ma_7 = daily_counts.rolling(window=7).mean()
    fig.add_trace(go.Scatter(
        x=ma_7.index,
        y=ma_7.values,
        mode='lines',
        name='7-day Moving Average',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='Review Volume Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Reviews',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def create_geographic_viz(df):
    """Create geographic visualizations."""
    # Country-wise analysis
    country_stats = df.groupby('user_country').agg({
        'rating': 'mean',
        'review_id': 'count'
    }).round(2)
    country_stats.columns = ['avg_rating', 'review_count']
    country_stats = country_stats.sort_values('review_count', ascending=False).head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=country_stats.index,
            y=country_stats['review_count'],
            title='Review Count by Country (Top 15)'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=country_stats.index,
            y=country_stats['avg_rating'],
            title='Average Rating by Country (Top 15)'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def create_sentiment_viz(df):
    """Create sentiment analysis visualizations."""
    if 'sentiment' not in df.columns:
        st.warning("Sentiment analysis not performed yet!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                    title='Sentiment Distribution',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment by rating
        fig = px.box(df, x='sentiment', y='rating', title='Rating Distribution by Sentiment',
                    color='sentiment',
                    color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment by language
    sentiment_lang = pd.crosstab(df['review_language'], df['sentiment'], normalize='index') * 100
    top_languages = df['review_language'].value_counts().head(8).index
    sentiment_lang_top = sentiment_lang.loc[top_languages]
    
    fig = px.bar(sentiment_lang_top, title='Sentiment Distribution by Language (Top 8)',
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'orange'})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📱 Multilingual App Reviews Analysis Dashboard</h1>
        <p>Interactive Machine Learning & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["📊 Dataset Overview", "🔍 Sentiment Analysis", "🤖 Rating Prediction", 
         "📈 Time Series Analysis", "🗺️ Geographic Analysis", "🎯 Model Comparison"]
    )
    
    # Dataset overview in sidebar
    st.sidebar.markdown("### 📋 Dataset Info")
    st.sidebar.metric("Total Reviews", len(df))
    st.sidebar.metric("Unique Apps", df['app_name'].nunique())
    st.sidebar.metric("Languages", df['review_language'].nunique())
    st.sidebar.metric("Countries", df['user_country'].nunique())
    st.sidebar.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    
    # Main content based on selection
    if analysis_type == "📊 Dataset Overview":
        st.header("📊 Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Reviews", f"{len(df):,}")
        with col2:
            st.metric("Unique Apps", df['app_name'].nunique())
        with col3:
            st.metric("Languages", df['review_language'].nunique())
        with col4:
            st.metric("Countries", df['user_country'].nunique())
        with col5:
            st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
        
        # EDA visualizations
        create_eda_visualizations(df)
        
        # Data sample
        st.subheader("📋 Data Sample")
        st.dataframe(df.head(10))
        
        # Basic statistics
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe())
    
    elif analysis_type == "🔍 Sentiment Analysis":
        st.header("🔍 Sentiment Analysis")
        
        # Perform sentiment analysis
        with st.spinner("Performing sentiment analysis..."):
            df_with_sentiment = perform_sentiment_analysis(df)
        
        # Train sentiment classification models
        with st.spinner("Training sentiment classification models..."):
            sentiment_models, sentiment_vectorizer = train_sentiment_classification_models(df_with_sentiment)
        
        # Model selection
        if sentiment_models:
            selected_sentiment_model = st.selectbox(
                "Select Sentiment Classification Model",
                list(sentiment_models.keys())
            )
            
            # Show model results
            create_sentiment_classification_viz(sentiment_models, selected_sentiment_model)
        
        # Sentiment visualizations
        create_sentiment_viz(df_with_sentiment)
    
    elif analysis_type == "🤖 Rating Prediction":
        st.header("🤖 Rating Prediction Models")
        
        # Train rating prediction models
        with st.spinner("Training rating prediction models..."):
            rating_models, rating_vectorizer = train_rating_prediction_models(df)
        
        if rating_models:
            # Model selection
            selected_rating_model = st.selectbox(
                "Select Rating Prediction Model",
                list(rating_models.keys())
            )
            
            # Show model results
            create_rating_prediction_viz(rating_models, selected_rating_model)
            
            # Model comparison
            st.subheader("📊 Model Comparison")
            comparison_df = pd.DataFrame({
                'Model': list(rating_models.keys()),
                'RMSE': [result['rmse'] for result in rating_models.values()],
                'MAE': [result['mae'] for result in rating_models.values()],
                'R² Score': [result['r2'] for result in rating_models.values()]
            })
            st.dataframe(comparison_df)
            
            # Interactive prediction
            st.subheader("🎯 Try Your Own Prediction")
            user_text = st.text_area("Enter review text:", "This app is amazing!")
            if st.button("Predict Rating"):
                try:
                    text_vector = rating_vectorizer.transform([user_text])
                    prediction = rating_models[selected_rating_model]['model'].predict(text_vector)[0]
                    st.success(f"Predicted Rating: {prediction:.2f} ⭐")
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    elif analysis_type == "📈 Time Series Analysis":
        st.header("📈 Time Series Analysis")
        create_time_series_viz(df)
        
        # Additional time series insights
        df['review_date'] = pd.to_datetime(df['review_date'])
        st.subheader("📊 Time Series Insights")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Date Range", f"{df['review_date'].dt.date.min()} to {df['review_date'].dt.date.max()}")
        with col2:
            daily_counts = df.groupby(df['review_date'].dt.date).size()
            st.metric("Peak Day", f"{daily_counts.idxmax()}")
        with col3:
            st.metric("Daily Average", f"{daily_counts.mean():.1f}")
    
    elif analysis_type == "🗺️ Geographic Analysis":
        st.header("🗺️ Geographic Analysis")
        create_geographic_viz(df)
        
        # Country selector for detailed analysis
        st.subheader("🔍 Country Deep Dive")
        selected_country = st.selectbox("Select Country", df['user_country'].unique())
        
        country_data = df[df['user_country'] == selected_country]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Reviews", len(country_data))
        with col2:
            st.metric("Avg Rating", f"{country_data['rating'].mean():.2f}")
        with col3:
            st.metric("Top App", country_data['app_name'].mode().iloc[0] if not country_data['app_name'].mode().empty else "N/A")
        with col4:
            st.metric("Languages", country_data['review_language'].nunique())
    
    elif analysis_type == "🎯 Model Comparison":
        st.header("🎯 Comprehensive Model Comparison")
        
        # Train all models
        with st.spinner("Training all models..."):
            df_with_sentiment = perform_sentiment_analysis(df)
            rating_models, _ = train_rating_prediction_models(df)
            sentiment_models, _ = train_sentiment_classification_models(df_with_sentiment)
        
        # Rating models comparison
        st.subheader("📊 Rating Prediction Models")
        if rating_models:
            rating_comparison = pd.DataFrame({
                'Model': list(rating_models.keys()),
                'RMSE': [result['rmse'] for result in rating_models.values()],
                'MAE': [result['mae'] for result in rating_models.values()],
                'R² Score': [result['r2'] for result in rating_models.values()]
            })
            
            # Best model highlighting
            best_rmse_idx = rating_comparison['RMSE'].idxmin()
            best_r2_idx = rating_comparison['R² Score'].idxmax()
            
            st.dataframe(rating_comparison.style.highlight_min(subset=['RMSE', 'MAE']).highlight_max(subset=['R² Score']))
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 Best RMSE: {rating_comparison.loc[best_rmse_idx, 'Model']} ({rating_comparison.loc[best_rmse_idx, 'RMSE']:.3f})")
            with col2:
                st.success(f"🏆 Best R²: {rating_comparison.loc[best_r2_idx, 'Model']} ({rating_comparison.loc[best_r2_idx, 'R² Score']:.3f})")
        
        # Sentiment models comparison
        st.subheader("🎭 Sentiment Classification Models")
        if sentiment_models:
            sentiment_comparison = pd.DataFrame({
                'Model': list(sentiment_models.keys()),
                'Accuracy': [result['accuracy'] for result in sentiment_models.values()]
            })
            
            best_accuracy_idx = sentiment_comparison['Accuracy'].idxmax()
            st.dataframe(sentiment_comparison.style.highlight_max(subset=['Accuracy']))
            st.success(f"🏆 Best Accuracy: {sentiment_comparison.loc[best_accuracy_idx, 'Model']} ({sentiment_comparison.loc[best_accuracy_idx, 'Accuracy']:.3f})")

if __name__ == "__main__":
    main()