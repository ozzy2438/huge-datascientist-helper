import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os
from dotenv import load_dotenv
import requests
import json
from openai import OpenAI
import httpx
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    http_client=httpx.Client()
)

# Available themes
THEMES = {
    "Dark Pro": {
        "bg_color": "#0A0A0F",
        "secondary_bg_color": "#1A1B26",
        "text_color": "#E2E8F0",
        "accent_color": "#7C3AED",
        "success_color": "#059669",
        "warning_color": "#D97706", 
        "error_color": "#DC2626",
        "chart_colors": ["#7C3AED", "#059669", "#D97706", "#DC2626", "#8B5CF6", "#EC4899"]
    }
}

def apply_theme(theme_name):
    """Apply selected theme with enhanced styling"""
    theme = THEMES[theme_name]
    st.markdown(f"""
        <style>
        /* Ana stil güncellemeleri */
        .main {{
            background-color: {theme["bg_color"]};
            color: {theme["text_color"]};
        }}
        
        /* Kart stili */
        .stCard {{
            background-color: {theme["secondary_bg_color"]};
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(99, 102, 241, 0.1);
        }}
        
        /* Buton stili */
        .stButton>button {{
            background-color: {theme["accent_color"]};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
        }}
        
        /* Metrik kartları */
        .css-1r6slb0 {{
            background-color: {theme["secondary_bg_color"]};
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid rgba(99, 102, 241, 0.1);
        }}
        
        /* Sekmeler */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {theme["secondary_bg_color"]};
            padding: 0.5rem;
            border-radius: 10px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            color: {theme["text_color"]};
            border-radius: 6px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {theme["accent_color"]};
        }}
        
        /* Grafik container */
        .plot-container {{
            background-color: {theme["secondary_bg_color"]};
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid rgba(99, 102, 241, 0.1);
        }}
        
        /* Sidebar */
        .css-1d391kg {{
            background-color: {theme["secondary_bg_color"]};
        }}
        </style>
    """, unsafe_allow_html=True)

def get_ai_insights(data_description, question):
    """Get AI insights using OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data science expert assistant. Provide clear, actionable insights and recommendations."},
                {"role": "user", "content": f"Data Description: {data_description}\n\nQuestion: {question}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI insights: {str(e)}"

def get_perplexity_response(query, df_description):
    """Get insights using Perplexity AI"""
    url = "https://api.perplexity.ai/chat/completions"
    
    analysis_prompt = f"""
    Based on this dataset and question, provide data collection recommendations:
    
    Dataset Info: {df_description}
    User Question: {query}
    
    Please analyze and suggest:
    1. Analyze the current dataset and identify gaps
    2. Find specific data sources (with direct URLs) that would complement this dataset
    3. Explain how each suggested data source would improve the model
    4. Provide integration steps for each data source
    
    Focus on finding real, accessible data sources that directly relate to this dataset.
    Include only working URLs and specific sources, not general platforms.
    """
    
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are a data source expert. Find specific, relevant data sources with working URLs."
            },
            {
                "role": "user", 
                "content": analysis_prompt
            }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 2048,
        "search_domain_filter": ["perplexity.ai"],
        "return_related_questions": True,
        "search_recency_filter": "month"
    }
    
    headers = {
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        results = response.json()
        
        # Sonuçları kategorize et
        formatted_results = {
            "AI Analysis": results['choices'][0]['message']['content'],
            "Data Sources": [],
            "Related Questions": results.get('related_questions', [])
        }
        
        return formatted_results
        
    except Exception as e:
        st.error(f"Perplexity API Error: {str(e)}")
        return {"error": str(e)}

def display_search_results(results):
    """Display enhanced search results with AI analysis and recommendations"""
    if isinstance(results, dict) and not results.get("error"):
        # AI Analysis
        if "AI Analysis" in results:
            st.subheader("🤖 AI Analiz & Öneriler")
            st.markdown(results["AI Analysis"])
            st.write("---")
        
        # Önerilen Kaynaklar
        if "Recommended Sources" in results:
            st.subheader("📚 Önerilen Kaynaklar")
            
            for category, sources in results["Recommended Sources"].items():
                with st.expander(f"📂 {category}"):
                    for source in sources:
                        st.markdown(f"""
                        #### [{source['name']}]({source['url']})
                        {source['description']}
                        """)
        
        # İlgili Sorular
        if "Related Questions" in results and results["Related Questions"]:
            st.subheader("❓ İlgili Sorular")
            for question in results["Related Questions"]:
                st.info(question)
        
        # Uygulama İpuçları
        st.subheader("💡 Uygulama İpuçları")
        st.markdown("""
        1. **Veri Toplama Stratejisi**
           - Önce pilot çalışma yapın
           - Veri kalitesi kontrollerini otomatikleştirin
           - Düzenli yedekleme planlayın
        
        2. **Veri Güvenliği**
           - Kişisel verileri anonimleştirin
           - Güvenli depolama kullanın
           - Erişim kontrolü uygulayın
        
        3. **Kalite Kontrol**
           - Veri doğrulama kuralları belirleyin
           - Düzenli denetimler yapın
           - Hata raporlama sistemi kurun
        """)
    else:
        st.warning("Sonuç bulunamadı.")

def create_visualization(df, viz_type, **kwargs):
    """Enhanced visualizations with modern styling"""
    theme = THEMES["Dark Pro"]
    
    template = {
        "layout": {
            "plot_bgcolor": theme["secondary_bg_color"],
            "paper_bgcolor": theme["secondary_bg_color"],
            "font": {"color": theme["text_color"]},
            "title": {"font": {"size": 24}},
            "margin": dict(t=40, l=40, r=40, b=40)
        }
    }
    
    if viz_type == "Distribution":
        fig = px.histogram(
            df, 
            x=kwargs.get('col'),
            marginal="violin",
            color_discrete_sequence=[theme["accent_color"]],
            template=template
        )
        
    elif viz_type == "Correlation":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        fig = px.imshow(
            df[numeric_cols].corr(),
            color_continuous_scale=['#EF4444', '#ffffff', '#6366F1'],
            template=template
        )
        
    elif viz_type == "Scatter Plot":
        fig = px.scatter(
            df,
            x=kwargs.get('x_col'),
            y=kwargs.get('y_col'),
            color_discrete_sequence=[theme["accent_color"]],
            template=template
        )
    
    elif viz_type == "Box Plot":
        fig = px.box(
            df,
            y=kwargs.get('col'),
            color_discrete_sequence=[theme["accent_color"]],
            template=template
        )
    
    # Stil güncellemeleri
    fig.update_layout(
        font_family="Arial",
        title_font_size=24,
        showlegend=True,
        legend=dict(
            bgcolor=theme["secondary_bg_color"],
            bordercolor=theme["accent_color"]
        ),
        plot_bgcolor=theme["secondary_bg_color"],
        paper_bgcolor=theme["bg_color"]
    )
    
    # Eksen stillerini güncelle
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.1)",
        zerolinecolor="rgba(255, 255, 255, 0.2)"
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.1)",
        zerolinecolor="rgba(255, 255, 255, 0.2)"
    )
    
    return fig

def preprocess_data(df):
    """Preprocess the dataframe with improved handling"""
    try:
        df = df.copy()
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Handle missing values
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                st.info(f"Filled missing values in '{col}' with median ({median_val:.2f})")
        
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                st.info(f"Filled missing values in '{col}' with mode ({mode_val})")
        
        # Encode categorical variables with warning for low variance
        le_dict = {}
        for col in categorical_cols:
            try:
                unique_values = df[col].nunique()
                if unique_values == 1:
                    st.warning(f"⚠️ Column '{col}' has only one unique value. Consider removing it.")
                
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = {
                    'encoder': le,
                    'classes': le.classes_.tolist(),
                    'unique_count': unique_values
                }
                
                # Add color coding based on unique value count
                if unique_values == 1:
                    st.markdown(f"🔴 '{col}': {unique_values} unique value")
                elif unique_values == 2:
                    st.markdown(f"🟡 '{col}': {unique_values} unique values")
                else:
                    st.markdown(f"🟢 '{col}': {unique_values} unique values")
                
            except Exception as e:
                st.error(f"Error encoding '{col}': {str(e)}")
        
        # Add feature recommendations
        st.subheader("📊 Feature Engineering Önerileri")
        
        # Binary columns
        binary_cols = [col for col in categorical_cols if le_dict.get(col, {}).get('unique_count') == 2]
        if binary_cols:
            st.info(f"💡 İkili değişkenler ({len(binary_cols)}): {', '.join(binary_cols)}")
            st.write("Bu değişkenler için one-hot encoding gerekli değil.")
        
        # Multi-class columns
        multi_cols = [col for col in categorical_cols if le_dict.get(col, {}).get('unique_count') > 2]
        if multi_cols:
            st.info(f"💡 Çok sınıflı değişkenler ({len(multi_cols)}): {', '.join(multi_cols)}")
            st.write("Bu değişkenler için one-hot encoding düşünülebilir.")
        
        # Low variance columns
        low_var_cols = [col for col in categorical_cols if le_dict.get(col, {}).get('unique_count') == 1]
        if low_var_cols:
            st.warning(f"⚠️ Tek değerli değişkenler ({len(low_var_cols)}): {', '.join(low_var_cols)}")
            st.write("Bu değişkenler modelden çıkarılabilir.")
        
        return df, le_dict
    
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return df, {}

def main():
    st.set_page_config(
        page_title="Data Science Studio",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Theme selection in sidebar
    with st.sidebar:
        st.title("🎨 Customization")
        selected_theme = st.selectbox("Select Theme", list(THEMES.keys()))
        apply_theme(selected_theme)
        
        st.title("📊 Data")
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])
        
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            processed_df, encoders = preprocess_data(df)
            
            # Veri kalitesi metriklerini göster
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Toplam Satır", df.shape[0])
            with col2:
                st.metric("Toplam Kolon", df.shape[1])
            with col3:
                st.metric("Kategorik Değişkenler", len(encoders))
            with col4:
                st.metric("Sayısal Değişkenler", len(df.select_dtypes(include=[np.number]).columns))
            
            # Main tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔍 Data Explorer",
                "📊 Visualization Studio",
                "🤖 AI Assistant",
                "📈 Model Development"
            ])
            
            with tab1:
                st.header("🔍 Data Explorer")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                st.subheader("Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Type': df.dtypes.astype(str),
                    'Missing': df.isnull().sum().values,
                    'Unique Values': df.nunique().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with tab2:
                st.header("📊 Visualization Studio")
                
                viz_type = st.selectbox(
                    "Select Visualization",
                    ["Distribution", "Correlation", "Scatter Plot", "Box Plot"]
                )
                
                # Visualization parametrelerini tanımla
                col = None
                x_col = None
                y_col = None
                
                if viz_type == "Distribution":
                    col = st.selectbox("Select Column", df.columns)
                elif viz_type == "Scatter Plot":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("Select X axis", df.columns)
                    with col2:
                        y_col = st.selectbox("Select Y axis", df.columns)
                elif viz_type == "Box Plot":
                    col = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns)
                
                # Görselleştirmeyi oluştur
                fig = create_visualization(df, viz_type, col=col, x_col=x_col, y_col=y_col)
                
                # Container içinde göster
                with st.container():
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.header("🤖 AI Assistant")
                
                # AI Query Section
                st.subheader("Ask AI Assistant")
                question = st.text_input("What would you like to know about your data?")
                
                if question:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("AI Insights")
                        data_description = f"Dataset with {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns)}"
                        ai_response = get_ai_insights(data_description, question)
                        st.markdown(f"""
                        <div style='background-color: #1A1B26; padding: 20px; border-radius: 10px; border-left: 4px solid #7C3AED;'>
                            <p style='color: #E2E8F0; margin: 0; white-space: pre-wrap;'>{ai_response}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("Related Resources")
                        web_results = get_perplexity_response(question, data_description)
                        display_search_results(web_results)
            
            with tab4:
                st.header("📈 Model Development")
                
                # Model Configuration
                st.subheader("Model Setup")
                target = st.selectbox("Select Target Variable", df.columns)
                features = st.multiselect(
                    "Select Features",
                    [col for col in df.columns if col != target],
                    default=[col for col in df.columns if col != target][:3]
                )
                
                if features and target:
                    # Model type selection
                    problem_type = st.radio(
                        "Select Problem Type",
                        ["Classification", "Regression"],
                        help="Choose based on your target variable type"
                    )
                    
                    # Prepare data
                    X = processed_df[features]
                    y = processed_df[target]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Train model
                    if st.button("Train Model"):
                        with st.spinner("Training model..."):
                            try:
                                if problem_type == "Classification":
                                    model = RandomForestClassifier(
                                        n_estimators=100,
                                        random_state=42,
                                        n_jobs=-1
                                    )
                                    st.info("Using Random Forest Classifier")
                                else:
                                    model = RandomForestRegressor(
                                        n_estimators=100,
                                        random_state=42,
                                        n_jobs=-1
                                    )
                                    st.info("Using Random Forest Regressor")
                                
                                with st.spinner("Training in progress..."):
                                    model.fit(X_train, y_train)
                                
                                # Model evaluation
                                train_score = model.score(X_train, y_train)
                                test_score = model.score(X_test, y_test)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Training Score", f"{train_score:.2%}")
                                with col2:
                                    st.metric("Testing Score", f"{test_score:.2%}")
                                
                                # Feature importance
                                importance_df = pd.DataFrame({
                                    'Feature': features,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                st.subheader("Feature Importance")
                                fig = px.bar(
                                    importance_df,
                                    x='Feature',
                                    y='Importance',
                                    title="Feature Importance Plot"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Error training model: {str(e)}")
                                st.info("Tip: Ensure your target variable is properly encoded for classification.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your file is a valid CSV file with proper formatting.")
    
    else:
        # Welcome screen
        st.title("🔬 Welcome to Data Science Studio")
        st.markdown("""
        ### Your Advanced Data Science Workspace
        
        Upload a dataset to:
        - 🔍 Explore and analyze your data
        - 📊 Create interactive visualizations
        - 🤖 Get AI-powered insights
        - 📈 Develop and evaluate models
        
        Get started by uploading a CSV file in the sidebar!
        """)

class ResearchAgent:
    def __init__(self, openai_client):
        self.openai = openai_client
    
    def search(self, query):
        # Web araması yap
        return get_perplexity_response(query)
    
    def analyze(self, data):
        # OpenAI ile analiz
        return self.get_ai_analysis(data)
    
    def recommend(self, current_data, search_results):
        # Öneriler oluştur
        return self.get_recommendations(current_data, search_results)

def get_enhanced_insights(query, df):
    """Get enhanced insights using CrewAI"""
    crew = DataAnalysisCrew(client, perplexity_client)
    results = crew.analyze_data(df, query)
    
    formatted_results = {
        "AI Analysis": results["Analysis"],
        "Data Sources": [
            {
                "name": source["name"],
                "url": source["url"],
                "description": source["description"],
                "data_type": source["type"],
                "integration_difficulty": source["difficulty"]
            }
            for source in results["Data Sources"]["Datasets"] + 
                         results["Data Sources"]["APIs"] + 
                         results["Data Sources"]["Web Sources"]
        ],
        "Integration Steps": results["Integration Steps"]
    }
    
    return formatted_results

class DataAnalysisCrew:
    def __init__(self, openai_client, perplexity_client):
        self.openai = openai_client
        self.perplexity = perplexity_client
        self.search_tool = DuckDuckGoSearchRun()
        
    def create_agents(self):
        # Data Analyst Agent - OpenAI kullanarak veri analizi yapar
        analyst = Agent(
            role='Data Analyst',
            goal='Analyze dataset and identify required additional data types',
            backstory='Expert in data analysis and feature engineering',
            tools=[self.openai_analysis],
            verbose=True
        )
        
        # Research Agent - Perplexity kullanarak veri kaynakları bulur
        researcher = Agent(
            role='Data Researcher',
            goal='Find relevant data sources and datasets',
            backstory='Expert in finding and evaluating data sources',
            tools=[self.perplexity_search, self.search_tool],
            verbose=True
        )
        
        # Integration Agent - Önerileri birleştirir ve formatlar
        integrator = Agent(
            role='Integration Specialist',
            goal='Combine analysis and research into actionable recommendations',
            backstory='Expert in data integration and recommendation systems',
            tools=[self.format_recommendations],
            verbose=True
        )
        
        return [analyst, researcher, integrator]
    
    def analyze_data(self, df, query):
        # Create tasks
        analysis_task = Task(
            description=f"Analyze this dataset: {df.describe()} and query: {query}",
            agent=self.create_agents()[0]
        )
        
        research_task = Task(
            description="Find specific data sources, datasets, and APIs related to the analysis",
            agent=self.create_agents()[1]
        )
        
        integration_task = Task(
            description="Combine analysis and research into formatted recommendations",
            agent=self.create_agents()[2]
        )
        
        # Create crew
        crew = Crew(
            agents=self.create_agents(),
            tasks=[analysis_task, research_task, integration_task],
            process=Process.sequential
        )
        
        # Execute crew
        result = crew.kickoff()
        
        return self.format_output(result)
    
    def format_output(self, result):
        """Format crew results into structured recommendations"""
        return {
            "Analysis": {
                "Current Data Analysis": result["analyst_insights"],
                "Required Additional Data": result["data_requirements"]
            },
            "Data Sources": {
                "Datasets": result["recommended_datasets"],
                "APIs": result["recommended_apis"],
                "Web Sources": result["web_sources"]
            },
            "Integration Steps": result["integration_steps"]
        }

if __name__ == "__main__":
    main()
