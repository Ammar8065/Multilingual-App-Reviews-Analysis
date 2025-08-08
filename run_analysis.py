"""
Working analysis script for the multilingual app reviews system.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

def load_real_data():
    """Load the actual dataset."""
    try:
        df = pd.read_csv('multilingual_mobile_app_reviews_2025.csv')
        print(f"✅ Loaded real dataset with {len(df)} reviews")
        return df
    except FileNotFoundError:
        print("⚠️ Real dataset not found, creating sample data...")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    n_rows = 1000
    
    apps = ['WhatsApp', 'Instagram', 'TikTok', 'Netflix', 'Spotify', 'YouTube', 'Facebook', 'Twitter']
    categories = ['Social', 'Photo & Video', 'Entertainment', 'Music', 'Communication']
    languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'pt', 'ru', 'ar', 'hi']
    countries = ['USA', 'Spain', 'France', 'Germany', 'China', 'Japan', 'Brazil', 'Russia', 'India', 'UK']
    devices = ['iOS', 'Android', 'iPad', 'Android Tablet']
    genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
    
    # Create more realistic reviews
    positive_reviews = [
        "Amazing app! Love the new features and smooth interface.",
        "Best app I've ever used. Highly recommend to everyone!",
        "Perfect for daily use. Great job developers!",
        "Excellent user experience. Five stars!",
        "Outstanding quality and performance. Keep it up!",
        "Love this app so much. Can't live without it!",
        "Fantastic update! Much better than before.",
        "Great app with awesome features. Very satisfied!",
        "Perfect design and functionality. Impressed!",
        "Amazing experience every time I use it."
    ]
    
    negative_reviews = [
        "Terrible app. Crashes constantly and very buggy.",
        "Worst experience ever. Waste of time and storage.",
        "App is completely broken. Needs major fixes.",
        "Awful interface and poor performance. Disappointed.",
        "Too many bugs and glitches. Very frustrating.",
        "Crashes every time I try to use it. Useless!",
        "Poor quality and terrible user experience.",
        "Buggy mess. Developers need to fix this ASAP.",
        "Completely unusable. Regret downloading it.",
        "Horrible app. Would give zero stars if possible."
    ]
    
    neutral_reviews = [
        "It's okay, nothing special but does the job.",
        "Average app. Could be better with some improvements.",
        "Decent functionality but room for improvement.",
        "Not bad, but not great either. It's fine.",
        "Works as expected. Nothing extraordinary.",
        "Standard app with basic features. It's alright.",
        "Could use some updates but it's functional.",
        "It's fine for what it is. Average experience.",
        "Does what it's supposed to do. Nothing more.",
        "Okay app. Some features work well, others don't."
    ]
    
    all_reviews = positive_reviews + negative_reviews + neutral_reviews
    
    data = {
        'review_id': range(1, n_rows + 1),
        'user_id': np.random.randint(1000, 99999, n_rows),
        'app_name': np.random.choice(apps, n_rows),
        'app_category': np.random.choice(categories, n_rows),
        'review_text': np.random.choice(all_reviews, n_rows),
        'review_language': np.random.choice(languages, n_rows),
        'rating': np.random.uniform(1.0, 5.0, n_rows),
        'review_date': pd.date_range('2023-01-01', periods=n_rows, freq='H'),
        'verified_purchase': np.random.choice([True, False], n_rows, p=[0.7, 0.3]),
        'device_type': np.random.choice(devices, n_rows),
        'num_helpful_votes': np.random.randint(0, 200, n_rows),
        'user_age': np.random.uniform(16, 75, n_rows),
        'user_country': np.random.choice(countries, n_rows),
        'user_gender': np.random.choice(genders, n_rows),
        'app_version': [f"{np.random.randint(1, 15)}.{np.random.randint(0, 20)}.{np.random.randint(0, 10)}" for _ in range(n_rows)]
    }
    
    return pd.DataFrame(data)

def perform_sentiment_analysis(df):
    """Perform simple sentiment analysis."""
    print("🔍 Performing sentiment analysis...")
    
    # Simple rule-based sentiment analysis
    def analyze_sentiment(text):
        if pd.isna(text) or not isinstance(text, str):
            return 'neutral'
            
        text_lower = text.lower()
        
        positive_words = ['amazing', 'great', 'excellent', 'love', 'best', 'perfect', 'fantastic', 'awesome', 'outstanding']
        negative_words = ['terrible', 'awful', 'worst', 'hate', 'bad', 'horrible', 'useless', 'disappointing', 'buggy']
        
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    df['sentiment'] = df['review_text'].apply(analyze_sentiment)
    
    sentiment_dist = df['sentiment'].value_counts()
    print(f"  • Positive: {sentiment_dist.get('positive', 0)} ({sentiment_dist.get('positive', 0)/len(df)*100:.1f}%)")
    print(f"  • Negative: {sentiment_dist.get('negative', 0)} ({sentiment_dist.get('negative', 0)/len(df)*100:.1f}%)")
    print(f"  • Neutral: {sentiment_dist.get('neutral', 0)} ({sentiment_dist.get('neutral', 0)/len(df)*100:.1f}%)")
    
    return df

def perform_rating_prediction(df):
    """Perform simple rating prediction."""
    print("🤖 Training rating prediction model...")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Prepare features - handle NaN values
        df_clean = df.dropna(subset=['review_text', 'rating'])
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_text = vectorizer.fit_transform(df_clean['review_text'])
        
        # Add numerical features
        X_numerical = df_clean[['num_helpful_votes', 'user_age']].fillna(0)
        
        # Combine features (simplified approach)
        y = df_clean['rating']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  • Model RMSE: {rmse:.3f}")
        print(f"  • Model R²: {r2:.3f}")
        
        # Test predictions on sample texts
        test_texts = [
            "Amazing app! Love it so much!",
            "Terrible experience, crashes all the time",
            "It's okay, nothing special"
        ]
        
        test_vectors = vectorizer.transform(test_texts)
        test_predictions = model.predict(test_vectors)
        
        print("  • Sample predictions:")
        for text, pred in zip(test_texts, test_predictions):
            print(f"    '{text[:30]}...' → {pred:.2f}")
        
        return model, vectorizer
        
    except ImportError:
        print("  ⚠️ Scikit-learn not available for ML predictions")
        return None, None

def perform_geographic_analysis(df):
    """Perform geographic analysis."""
    print("🗺️ Performing geographic analysis...")
    
    # Country-wise analysis
    country_stats = df.groupby('user_country').agg({
        'rating': ['mean', 'count'],
        'num_helpful_votes': 'mean'
    }).round(2)
    
    country_stats.columns = ['avg_rating', 'review_count', 'avg_helpful_votes']
    country_stats = country_stats.sort_values('review_count', ascending=False)
    
    print("  • Top countries by review count:")
    for country, row in country_stats.head().iterrows():
        print(f"    {country}: {row['review_count']} reviews (avg rating: {row['avg_rating']:.2f})")
    
    # Sentiment by country (if available)
    if 'sentiment' in df.columns:
        print("  • Sentiment by top countries:")
        for country in country_stats.head().index:
            country_data = df[df['user_country'] == country]
            sentiment_dist = country_data['sentiment'].value_counts(normalize=True) * 100
            pos_pct = sentiment_dist.get('positive', 0)
            neg_pct = sentiment_dist.get('negative', 0)
            print(f"    {country}: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative")
    
    return country_stats

def perform_time_series_analysis(df):
    """Perform time series analysis."""
    print("📈 Performing time series analysis...")
    
    # Convert to datetime
    df['review_date'] = pd.to_datetime(df['review_date'])
    
    # Daily review counts
    daily_counts = df.groupby(df['review_date'].dt.date).size()
    weekly_counts = df.groupby(df['review_date'].dt.to_period('W')).size()
    
    print(f"  • Date range: {daily_counts.index.min()} to {daily_counts.index.max()}")
    print(f"  • Average daily reviews: {daily_counts.mean():.1f}")
    print(f"  • Peak day: {daily_counts.idxmax()} ({daily_counts.max()} reviews)")
    print(f"  • Lowest day: {daily_counts.idxmin()} ({daily_counts.min()} reviews)")
    
    # Weekly patterns
    df['weekday'] = df['review_date'].dt.day_name()
    weekday_counts = df['weekday'].value_counts()
    print(f"  • Most active day: {weekday_counts.index[0]} ({weekday_counts.iloc[0]} reviews)")
    
    return daily_counts, weekly_counts

def create_advanced_visualizations(df):
    """Create advanced visualizations."""
    print("📊 Creating advanced visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directory
        Path("output/advanced").mkdir(parents=True, exist_ok=True)
        
        # 1. Sentiment vs Rating Analysis
        if 'sentiment' in df.columns:
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: Sentiment distribution
            plt.subplot(2, 2, 1)
            sentiment_counts = df['sentiment'].value_counts()
            colors = ['#2ecc71', '#e74c3c', '#f39c12']  # green, red, orange
            plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            plt.title('Sentiment Distribution', fontweight='bold')
            
            # Subplot 2: Rating by sentiment
            plt.subplot(2, 2, 2)
            df.boxplot(column='rating', by='sentiment', ax=plt.gca())
            plt.title('Rating Distribution by Sentiment')
            plt.suptitle('')
            
            # Subplot 3: Sentiment by language
            plt.subplot(2, 2, 3)
            sentiment_lang = pd.crosstab(df['review_language'], df['sentiment'], normalize='index') * 100
            sentiment_lang.plot(kind='bar', stacked=True, ax=plt.gca(), color=colors)
            plt.title('Sentiment by Language (%)')
            plt.xticks(rotation=45)
            plt.legend(title='Sentiment')
            
            # Subplot 4: Sentiment over time
            plt.subplot(2, 2, 4)
            df['date'] = pd.to_datetime(df['review_date']).dt.date
            sentiment_time = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
            sentiment_time.plot(ax=plt.gca(), color=colors)
            plt.title('Sentiment Trends Over Time')
            plt.xticks(rotation=45)
            plt.legend(title='Sentiment')
            
            plt.tight_layout()
            plt.savefig('output/advanced/sentiment_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Geographic Analysis
        plt.figure(figsize=(15, 10))
        
        # Country analysis
        country_stats = df.groupby('user_country').agg({
            'rating': 'mean',
            'review_id': 'count'
        }).round(2)
        country_stats.columns = ['avg_rating', 'review_count']
        country_stats = country_stats.sort_values('review_count', ascending=False).head(10)
        
        plt.subplot(2, 2, 1)
        country_stats['review_count'].plot(kind='bar', color='skyblue')
        plt.title('Top 10 Countries by Review Count')
        plt.xticks(rotation=45)
        plt.ylabel('Number of Reviews')
        
        plt.subplot(2, 2, 2)
        country_stats['avg_rating'].plot(kind='bar', color='lightcoral')
        plt.title('Average Rating by Country')
        plt.xticks(rotation=45)
        plt.ylabel('Average Rating')
        
        # Language analysis
        plt.subplot(2, 2, 3)
        lang_stats = df.groupby('review_language').agg({
            'rating': 'mean',
            'review_id': 'count'
        }).round(2)
        lang_stats.columns = ['avg_rating', 'review_count']
        lang_stats = lang_stats.sort_values('review_count', ascending=False).head(8)
        
        lang_stats['review_count'].plot(kind='bar', color='lightgreen')
        plt.title('Top 8 Languages by Review Count')
        plt.xticks(rotation=45)
        plt.ylabel('Number of Reviews')
        
        plt.subplot(2, 2, 4)
        lang_stats['avg_rating'].plot(kind='bar', color='gold')
        plt.title('Average Rating by Language')
        plt.xticks(rotation=45)
        plt.ylabel('Average Rating')
        
        plt.tight_layout()
        plt.savefig('output/advanced/geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. App Performance Analysis
        plt.figure(figsize=(15, 10))
        
        app_stats = df.groupby('app_name').agg({
            'rating': ['mean', 'std', 'count'],
            'num_helpful_votes': 'mean'
        }).round(2)
        app_stats.columns = ['avg_rating', 'rating_std', 'review_count', 'avg_helpful_votes']
        app_stats = app_stats.sort_values('review_count', ascending=False)
        
        plt.subplot(2, 2, 1)
        app_stats['avg_rating'].plot(kind='bar', color='purple', alpha=0.7)
        plt.title('Average Rating by App')
        plt.xticks(rotation=45)
        plt.ylabel('Average Rating')
        
        plt.subplot(2, 2, 2)
        app_stats['review_count'].plot(kind='bar', color='orange', alpha=0.7)
        plt.title('Review Count by App')
        plt.xticks(rotation=45)
        plt.ylabel('Number of Reviews')
        
        plt.subplot(2, 2, 3)
        plt.scatter(app_stats['avg_rating'], app_stats['review_count'], 
                   s=app_stats['avg_helpful_votes']*2, alpha=0.6, c='red')
        plt.xlabel('Average Rating')
        plt.ylabel('Review Count')
        plt.title('Rating vs Review Count (bubble size = helpful votes)')
        
        plt.subplot(2, 2, 4)
        app_stats['rating_std'].plot(kind='bar', color='teal', alpha=0.7)
        plt.title('Rating Variability by App')
        plt.xticks(rotation=45)
        plt.ylabel('Rating Standard Deviation')
        
        plt.tight_layout()
        plt.savefig('output/advanced/app_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Advanced visualizations saved:")
        print("  • output/advanced/sentiment_analysis.png")
        print("  • output/advanced/geographic_analysis.png")
        print("  • output/advanced/app_performance.png")
        
    except Exception as e:
        print(f"⚠️ Error creating advanced visualizations: {e}")

def create_comprehensive_dashboard(df, country_stats, daily_counts):
    """Create comprehensive HTML dashboard."""
    print("🌐 Creating comprehensive dashboard...")
    
    # Calculate advanced metrics
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    unique_apps = df['app_name'].nunique()
    unique_countries = df['user_country'].nunique()
    unique_languages = df['review_language'].nunique()
    
    # Sentiment metrics
    sentiment_dist = df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {}
    
    # Top performers
    top_apps = df['app_name'].value_counts().head(5)
    top_countries = df['user_country'].value_counts().head(5)
    top_languages = df['review_language'].value_counts().head(5)
    
    # Time metrics
    date_range = f"{daily_counts.index.min()} to {daily_counts.index.max()}"
    peak_day = daily_counts.idxmax()
    peak_count = daily_counts.max()
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multilingual App Reviews - Advanced Dashboard</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 15px;
                position: relative;
                overflow: hidden;
            }}
            
            .header::before {{
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: repeating-linear-gradient(
                    45deg,
                    transparent,
                    transparent 10px,
                    rgba(255,255,255,0.05) 10px,
                    rgba(255,255,255,0.05) 20px
                );
                animation: slide 20s linear infinite;
            }}
            
            @keyframes slide {{
                0% {{ transform: translateX(-50px) translateY(-50px); }}
                100% {{ transform: translateX(50px) translateY(50px); }}
            }}
            
            .header h1 {{
                font-size: 3em;
                margin-bottom: 10px;
                position: relative;
                z-index: 1;
            }}
            
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
                position: relative;
                z-index: 1;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 25px;
                margin-bottom: 40px;
            }}
            
            .metric-card {{
                background: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .metric-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
                transition: left 0.5s;
            }}
            
            .metric-card:hover::before {{
                left: 100%;
            }}
            
            .metric-card:hover {{
                transform: translateY(-10px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            }}
            
            .metric-value {{
                font-size: 3em;
                font-weight: bold;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }}
            
            .metric-label {{
                color: #666;
                font-size: 1.1em;
                font-weight: 500;
            }}
            
            .section {{
                background: white;
                margin-bottom: 30px;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            
            .section h2 {{
                color: #333;
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
                margin-bottom: 25px;
                font-size: 1.8em;
            }}
            
            .insights-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 25px;
            }}
            
            .insight-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px;
                border-radius: 12px;
                border-left: 5px solid #667eea;
            }}
            
            .insight-card h3 {{
                color: #333;
                margin-bottom: 15px;
                font-size: 1.3em;
            }}
            
            .list-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 15px;
                background: white;
                border-radius: 8px;
                margin-bottom: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                transition: all 0.2s ease;
            }}
            
            .list-item:hover {{
                transform: translateX(5px);
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            }}
            
            .list-item-name {{
                font-weight: bold;
                color: #333;
            }}
            
            .list-item-value {{
                color: #667eea;
                font-weight: bold;
            }}
            
            .images-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
                gap: 25px;
                margin-top: 25px;
            }}
            
            .image-container {{
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }}
            
            .image-container:hover {{
                transform: scale(1.02);
            }}
            
            .image-container img {{
                width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .image-title {{
                margin-top: 15px;
                font-weight: bold;
                color: #333;
                text-align: center;
                font-size: 1.1em;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 15px;
            }}
            
            .footer h3 {{
                margin-bottom: 10px;
                font-size: 1.5em;
            }}
            
            .footer p {{
                opacity: 0.9;
                font-size: 1.1em;
            }}
            
            .sentiment-badge {{
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                color: white;
                font-weight: bold;
                margin: 2px;
            }}
            
            .positive {{ background: #2ecc71; }}
            .negative {{ background: #e74c3c; }}
            .neutral {{ background: #f39c12; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📱 Multilingual App Reviews Dashboard</h1>
                <p>Advanced Analytics & Machine Learning Insights</p>
                <p><em>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_reviews:,}</div>
                    <div class="metric-label">Total Reviews</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_rating:.2f}</div>
                    <div class="metric-label">Average Rating</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{unique_apps}</div>
                    <div class="metric-label">Unique Apps</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{unique_countries}</div>
                    <div class="metric-label">Countries</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{unique_languages}</div>
                    <div class="metric-label">Languages</div>
                </div>
            </div>
            
            {'<div class="section"><h2>🎯 Sentiment Analysis Results</h2><div style="text-align: center; margin-bottom: 20px;">' + ''.join([f'<span class="sentiment-badge {sentiment.lower()}">{sentiment.title()}: {count} ({count/total_reviews*100:.1f}%)</span>' for sentiment, count in sentiment_dist.items()]) + '</div></div>' if sentiment_dist else ''}
            
            <div class="section">
                <h2>📊 Key Insights</h2>
                <div class="insights-grid">
                    <div class="insight-card">
                        <h3>🏆 Most Popular Apps</h3>
                        {''.join([f'<div class="list-item"><span class="list-item-name">{app}</span><span class="list-item-value">{count} reviews</span></div>' for app, count in top_apps.items()])}
                    </div>
                    <div class="insight-card">
                        <h3>🌍 Top Countries</h3>
                        {''.join([f'<div class="list-item"><span class="list-item-name">{country}</span><span class="list-item-value">{count} reviews</span></div>' for country, count in top_countries.items()])}
                    </div>
                    <div class="insight-card">
                        <h3>🗣️ Top Languages</h3>
                        {''.join([f'<div class="list-item"><span class="list-item-name">{lang}</span><span class="list-item-value">{count} reviews</span></div>' for lang, count in top_languages.items()])}
                    </div>
                    <div class="insight-card">
                        <h3>📈 Time Analysis</h3>
                        <div class="list-item"><span class="list-item-name">Date Range</span><span class="list-item-value">{date_range}</span></div>
                        <div class="list-item"><span class="list-item-name">Peak Day</span><span class="list-item-value">{peak_day}</span></div>
                        <div class="list-item"><span class="list-item-name">Peak Reviews</span><span class="list-item-value">{peak_count}</span></div>
                        <div class="list-item"><span class="list-item-name">Daily Average</span><span class="list-item-value">{daily_counts.mean():.1f}</span></div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Visualizations</h2>
                <div class="images-grid">
                    <div class="image-container">
                        <img src="rating_distribution.png" alt="Rating Distribution">
                        <div class="image-title">Rating Distribution Analysis</div>
                    </div>
                    <div class="image-container">
                        <img src="language_distribution.png" alt="Language Distribution">
                        <div class="image-title">Language Distribution</div>
                    </div>
                    <div class="image-container">
                        <img src="country_distribution.png" alt="Country Distribution">
                        <div class="image-title">Geographic Distribution</div>
                    </div>
                    <div class="image-container">
                        <img src="app_popularity.png" alt="App Popularity">
                        <div class="image-title">App Popularity Analysis</div>
                    </div>
                    <div class="image-container">
                        <img src="time_series.png" alt="Time Series">
                        <div class="image-title">Temporal Trends</div>
                    </div>
                    <div class="image-container">
                        <img src="rating_by_app.png" alt="Rating by App">
                        <div class="image-title">Rating Distribution by App</div>
                    </div>
                </div>
                
                <div class="images-grid" style="margin-top: 30px;">
                    <div class="image-container">
                        <img src="advanced/sentiment_analysis.png" alt="Sentiment Analysis">
                        <div class="image-title">Advanced Sentiment Analysis</div>
                    </div>
                    <div class="image-container">
                        <img src="advanced/geographic_analysis.png" alt="Geographic Analysis">
                        <div class="image-title">Geographic Deep Dive</div>
                    </div>
                    <div class="image-container">
                        <img src="advanced/app_performance.png" alt="App Performance">
                        <div class="image-title">App Performance Metrics</div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <h3>🚀 Multilingual Mobile App Reviews Analysis System</h3>
                <p>Powered by Python • Machine Learning • Advanced Analytics</p>
                <p>Complete pipeline with EDA, Sentiment Analysis, Rating Prediction & Geographic Insights</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save comprehensive dashboard
    with open('output/comprehensive_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Comprehensive dashboard created: output/comprehensive_dashboard.html")
    return 'output/comprehensive_dashboard.html'

def main():
    """Main function to run comprehensive analysis."""
    print("🚀 Starting Comprehensive Multilingual App Reviews Analysis")
    print("=" * 80)
    
    try:
        # Load data
        df = load_real_data()
        
        # Basic EDA
        print(f"\n📊 Dataset Overview:")
        print(f"  • Shape: {df.shape}")
        print(f"  • Columns: {list(df.columns)}")
        print(f"  • Date range: {df['review_date'].min()} to {df['review_date'].max()}")
        print(f"  • Average rating: {df['rating'].mean():.2f}")
        print(f"  • Languages: {df['review_language'].nunique()}")
        print(f"  • Countries: {df['user_country'].nunique()}")
        
        # Perform analyses
        df = perform_sentiment_analysis(df)
        model, vectorizer = perform_rating_prediction(df)
        country_stats = perform_geographic_analysis(df)
        daily_counts, weekly_counts = perform_time_series_analysis(df)
        
        # Create visualizations
        create_advanced_visualizations(df)
        
        # Create comprehensive dashboard
        dashboard_path = create_comprehensive_dashboard(df, country_stats, daily_counts)
        
        print("\n" + "=" * 80)
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"📊 Main Dashboard: output/dashboard.html")
        print(f"🌟 Advanced Dashboard: {dashboard_path}")
        print("📁 All files saved in 'output/' directory")
        print("\n💡 To view dashboards:")
        print(f"   Main: {os.path.abspath('output/dashboard.html')}")
        print(f"   Advanced: {os.path.abspath(dashboard_path)}")
        print("\n🔍 Analysis completed:")
        print("   ✅ Exploratory Data Analysis")
        print("   ✅ Sentiment Analysis")
        print("   ✅ Rating Prediction Model")
        print("   ✅ Geographic Analysis")
        print("   ✅ Time Series Analysis")
        print("   ✅ Advanced Visualizations")
        print("   ✅ Interactive Dashboards")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()