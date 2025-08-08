"""
Simple demo script to run analysis and create dashboard.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Sample data similar to your dataset
    n_rows = 500
    
    apps = ['WhatsApp', 'Instagram', 'TikTok', 'Netflix', 'Spotify']
    categories = ['Social', 'Photo & Video', 'Entertainment', 'Music']
    languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'pt', 'ru']
    countries = ['USA', 'Spain', 'France', 'Germany', 'China', 'Japan', 'Brazil', 'Russia']
    devices = ['iOS', 'Android', 'iPad', 'Android Tablet']
    genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
    
    # Create sample reviews
    positive_reviews = [
        "Great app, love it!", "Amazing experience!", "Perfect app!",
        "Excellent features", "Highly recommend", "Love this app so much",
        "Best app ever", "Outstanding quality", "Fantastic user interface"
    ]
    
    negative_reviews = [
        "Terrible app", "Crashes all the time", "Worst experience ever",
        "Buggy and slow", "Hate this app", "Complete waste of time",
        "Awful interface", "Too many ads", "Poor quality"
    ]
    
    neutral_reviews = [
        "It's okay", "Average app", "Nothing special", "Could be better",
        "Decent app", "Not bad", "It works fine", "Standard features"
    ]
    
    all_reviews = positive_reviews + negative_reviews + neutral_reviews
    
    data = {
        'review_id': range(1, n_rows + 1),
        'user_id': np.random.randint(1000, 9999, n_rows),
        'app_name': np.random.choice(apps, n_rows),
        'app_category': np.random.choice(categories, n_rows),
        'review_text': np.random.choice(all_reviews, n_rows),
        'review_language': np.random.choice(languages, n_rows),
        'rating': np.random.uniform(1.0, 5.0, n_rows),
        'review_date': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
        'verified_purchase': np.random.choice([True, False], n_rows),
        'device_type': np.random.choice(devices, n_rows),
        'num_helpful_votes': np.random.randint(0, 100, n_rows),
        'user_age': np.random.uniform(18, 70, n_rows),
        'user_country': np.random.choice(countries, n_rows),
        'user_gender': np.random.choice(genders, n_rows),
        'app_version': [f"1.{np.random.randint(0, 10)}.{np.random.randint(0, 10)}" for _ in range(n_rows)]
    }
    
    return pd.DataFrame(data)

def run_basic_analysis():
    """Run basic analysis and create visualizations."""
    print("🚀 Starting Multilingual App Reviews Analysis Demo")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample data...")
    df = create_sample_data()
    print(f"✅ Created dataset with {len(df)} reviews")
    
    # Basic EDA
    print("\n📈 Basic Exploratory Data Analysis:")
    print(f"  • Total Reviews: {len(df):,}")
    print(f"  • Unique Apps: {df['app_name'].nunique()}")
    print(f"  • Languages: {df['review_language'].nunique()}")
    print(f"  • Countries: {df['user_country'].nunique()}")
    print(f"  • Average Rating: {df['rating'].mean():.2f}")
    print(f"  • Date Range: {df['review_date'].min().date()} to {df['review_date'].max().date()}")
    
    # Language distribution
    print("\n🌍 Language Distribution:")
    lang_dist = df['review_language'].value_counts()
    for lang, count in lang_dist.head().items():
        print(f"  • {lang}: {count} reviews ({count/len(df)*100:.1f}%)")
    
    # Country distribution
    print("\n🗺️ Country Distribution:")
    country_dist = df['user_country'].value_counts()
    for country, count in country_dist.head().items():
        print(f"  • {country}: {count} reviews ({count/len(df)*100:.1f}%)")
    
    # App popularity
    print("\n📱 App Popularity:")
    app_dist = df['app_name'].value_counts()
    for app, count in app_dist.items():
        avg_rating = df[df['app_name'] == app]['rating'].mean()
        print(f"  • {app}: {count} reviews (avg rating: {avg_rating:.2f})")
    
    # Rating distribution
    print("\n⭐ Rating Distribution:")
    rating_ranges = [
        (1.0, 2.0, "Very Poor"),
        (2.0, 3.0, "Poor"),
        (3.0, 4.0, "Average"),
        (4.0, 5.0, "Good"),
        (5.0, 5.1, "Excellent")
    ]
    
    for min_r, max_r, label in rating_ranges:
        count = len(df[(df['rating'] >= min_r) & (df['rating'] < max_r)])
        if min_r == 5.0:  # Handle exact 5.0 ratings
            count = len(df[df['rating'] == 5.0])
        print(f"  • {label} ({min_r}-{max_r if max_r <= 5.0 else '5.0'}): {count} reviews ({count/len(df)*100:.1f}%)")
    
    return df

def create_simple_visualizations(df):
    """Create simple visualizations using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("\n📊 Creating visualizations...")
        
        # Create output directory
        Path("output").mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Rating distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['rating'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of App Ratings', fontsize=16, fontweight='bold')
        plt.xlabel('Rating')
        plt.ylabel('Number of Reviews')
        plt.axvline(df['rating'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["rating"].mean():.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Language distribution
        plt.figure(figsize=(12, 6))
        lang_counts = df['review_language'].value_counts()
        lang_counts.plot(kind='bar', color='lightcoral')
        plt.title('Review Count by Language', fontsize=16, fontweight='bold')
        plt.xlabel('Language')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/language_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Country distribution
        plt.figure(figsize=(12, 6))
        country_counts = df['user_country'].value_counts()
        country_counts.plot(kind='bar', color='lightgreen')
        plt.title('Review Count by Country', fontsize=16, fontweight='bold')
        plt.xlabel('Country')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/country_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. App popularity
        plt.figure(figsize=(10, 6))
        app_counts = df['app_name'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(app_counts)))
        plt.pie(app_counts.values, labels=app_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('App Popularity Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('output/app_popularity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Time series of reviews
        plt.figure(figsize=(14, 6))
        daily_counts = df.groupby(df['review_date'].dt.date).size()
        plt.plot(daily_counts.index, daily_counts.values, color='purple', linewidth=2)
        plt.title('Daily Review Volume Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Rating by app
        plt.figure(figsize=(12, 8))
        df.boxplot(column='rating', by='app_name', ax=plt.gca())
        plt.title('Rating Distribution by App', fontsize=16, fontweight='bold')
        plt.suptitle('')  # Remove default title
        plt.xlabel('App Name')
        plt.ylabel('Rating')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('output/rating_by_app.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations saved to 'output/' directory:")
        print("  • rating_distribution.png")
        print("  • language_distribution.png") 
        print("  • country_distribution.png")
        print("  • app_popularity.png")
        print("  • time_series.png")
        print("  • rating_by_app.png")
        
    except ImportError:
        print("⚠️ Matplotlib not available for visualizations")

def create_simple_dashboard_html(df):
    """Create a simple HTML dashboard."""
    print("\n🌐 Creating HTML dashboard...")
    
    # Calculate key metrics
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    unique_apps = df['app_name'].nunique()
    unique_countries = df['user_country'].nunique()
    unique_languages = df['review_language'].nunique()
    
    # Top apps
    top_apps = df['app_name'].value_counts().head(5)
    top_countries = df['user_country'].value_counts().head(5)
    top_languages = df['review_language'].value_counts().head(5)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multilingual App Reviews Dashboard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }}
            .metric-card:hover {{
                transform: translateY(-2px);
            }}
            .metric-value {{
                font-size: 2.5em;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 10px;
            }}
            .metric-label {{
                color: #666;
                font-size: 1.1em;
            }}
            .section {{
                background: white;
                margin-bottom: 30px;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #333;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .top-list {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }}
            .list-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                margin-bottom: 10px;
                border-left: 4px solid #667eea;
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
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .image-container {{
                text-align: center;
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
            }}
            .image-title {{
                margin-top: 15px;
                font-weight: bold;
                color: #333;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: #666;
                background: white;
                border-radius: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📱 Multilingual Mobile App Reviews Dashboard</h1>
            <p>Comprehensive Analysis of App Store Reviews</p>
            <p><em>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </div>
        
        <div class="metrics">
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
        
        <div class="section">
            <h2>📊 Top Performers</h2>
            <div class="top-list">
                <div>
                    <h3>🏆 Most Reviewed Apps</h3>
                    {''.join([f'<div class="list-item"><span class="list-item-name">{app}</span><span class="list-item-value">{count} reviews</span></div>' for app, count in top_apps.items()])}
                </div>
                <div>
                    <h3>🌍 Top Countries</h3>
                    {''.join([f'<div class="list-item"><span class="list-item-name">{country}</span><span class="list-item-value">{count} reviews</span></div>' for country, count in top_countries.items()])}
                </div>
                <div>
                    <h3>🗣️ Top Languages</h3>
                    {''.join([f'<div class="list-item"><span class="list-item-name">{lang}</span><span class="list-item-value">{count} reviews</span></div>' for lang, count in top_languages.items()])}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Visualizations</h2>
            <div class="images-grid">
                <div class="image-container">
                    <img src="rating_distribution.png" alt="Rating Distribution">
                    <div class="image-title">Rating Distribution</div>
                </div>
                <div class="image-container">
                    <img src="language_distribution.png" alt="Language Distribution">
                    <div class="image-title">Language Distribution</div>
                </div>
                <div class="image-container">
                    <img src="country_distribution.png" alt="Country Distribution">
                    <div class="image-title">Country Distribution</div>
                </div>
                <div class="image-container">
                    <img src="app_popularity.png" alt="App Popularity">
                    <div class="image-title">App Popularity</div>
                </div>
                <div class="image-container">
                    <img src="time_series.png" alt="Time Series">
                    <div class="image-title">Daily Review Volume</div>
                </div>
                <div class="image-container">
                    <img src="rating_by_app.png" alt="Rating by App">
                    <div class="image-title">Rating Distribution by App</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🚀 <strong>Multilingual Mobile App Reviews Analysis System</strong></p>
            <p>Built with Python • Pandas • Matplotlib • Machine Learning</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    with open('output/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML dashboard created: output/dashboard.html")
    return 'output/dashboard.html'

def main():
    """Main function to run the demo."""
    try:
        # Run basic analysis
        df = run_basic_analysis()
        
        # Create visualizations
        create_simple_visualizations(df)
        
        # Create HTML dashboard
        dashboard_path = create_simple_dashboard_html(df)
        
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"📊 Dashboard: {dashboard_path}")
        print("📁 All files saved in 'output/' directory")
        print("\n💡 To view the dashboard:")
        print(f"   Open: {os.path.abspath(dashboard_path)}")
        print("   Or run: python -m http.server 8000")
        print("   Then visit: http://localhost:8000/output/dashboard.html")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()