#!/usr/bin/env python3
"""
Victor AI Data Analysis Dashboard - Streamlit App

Interactive dashboard for data analysis with AI-powered insights.
"""

import sys
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.agent.orchestrator_factory import create_orchestrator
from victor.config.settings import Settings
from src.analyzer import DataAnalyzer
from src.visualizer import DataVisualizer
from src.insight_generator import InsightGenerator


# Page configuration
st.set_page_config(
    page_title="Victor AI Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .insight-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'orchestrator' not in st.session_state:
        settings = Settings()
        st.session_state.orchestrator = create_orchestrator(settings)

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer(st.session_state.orchestrator)

    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = DataVisualizer()

    if 'insight_generator' not in st.session_state:
        st.session_state.insight_generator = InsightGenerator(st.session_state.orchestrator)

    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False


def load_data(file):
    """Load data from uploaded file."""
    try:
        # Determine file type
        file_extension = Path(file.name).suffix.lower()

        if file_extension == '.csv':
            df = pd.read_csv(file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file)
        elif file_extension == '.json':
            df = pd.read_json(file)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def display_data_overview(df):
    """Display data overview section."""
    st.subheader("📋 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{len(df):,}")

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Display data types
    st.subheader("Column Types")
    col_types = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.notnull().sum().values,
        'Null': df.isnull().sum().values,
    })
    st.dataframe(col_types, use_container_width=True)

    # Display first rows
    st.subheader("Sample Data")
    st.dataframe(df.head(), use_container_width=True)


def display_statistical_analysis(df):
    """Display statistical analysis."""
    st.subheader("📊 Statistical Analysis")

    # Numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    if numerical_cols:
        with st.expander("Descriptive Statistics", expanded=True):
            stats = df[numerical_cols].describe()
            st.dataframe(stats, use_container_width=True)

        # Correlation matrix
        if len(numerical_cols) > 1:
            with st.expander("Correlation Matrix"):
                corr_matrix = df[numerical_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)

    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        with st.expander("Categorical Columns"):
            for col in categorical_cols[:5]:  # Limit to 5
                st.write(f"**{col}**")
                value_counts = df[col].value_counts().head(10)
                st.dataframe(value_counts, use_container_width=True)


def display_visualizations(df):
    """Display interactive visualizations."""
    st.subheader("📈 Visualizations")

    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Let user select visualization type
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart", "Heatmap"]
    )

    if viz_type == "Histogram" and numerical_cols:
        col = st.selectbox("Select Column", numerical_cols)
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Box Plot" and numerical_cols:
        col = st.selectbox("Select Column", numerical_cols)
        fig = px.box(df, y=col, title=f"Box Plot of {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Scatter Plot" and len(numerical_cols) >= 2:
        x_col = st.selectbox("X Axis", numerical_cols)
        y_col = st.selectbox("Y Axis", [c for c in numerical_cols if c != x_col])
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Bar Chart" and categorical_cols:
        col = st.selectbox("Select Column", categorical_cols)
        value_counts = df[col].value_counts().head(15)
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Count by {col}",
            labels={'x': col, 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Line Chart" and numerical_cols:
        col = st.selectbox("Select Column", numerical_cols)
        fig = px.line(df, y=col, title=f"{col} Over Time")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Heatmap" and len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_insights(df):
    """Display AI-generated insights."""
    st.subheader("🤖 AI-Powered Insights")

    if not st.session_state.analysis_complete:
        with st.spinner("Generating insights..."):
            insights = st.session_state.insight_generator.generate(df)
            st.session_state.insights = insights
            st.session_state.analysis_complete = True
    else:
        insights = st.session_state.insights

    # Display insights
    for insight in insights.get('key_insights', []):
        st.markdown(f"""
        <div class="insight-card">
            <h4>{insight['title']}</h4>
            <p>{insight['description']}</p>
        </div>
        """, unsafe_allow_html=True)


def natural_language_query(df):
    """Natural language query interface."""
    st.subheader("💬 Ask Questions About Your Data")

    query = st.text_input(
        "Ask a question (e.g., 'What is the average price?')",
        placeholder="Type your question here...",
        key="nl_query"
    )

    if st.button("Analyze", key="analyze_query"):
        if query:
            with st.spinner("Analyzing..."):
                result = st.session_state.analyzer.query(df, query)

                if result:
                    st.success("Result:")

                    if 'answer' in result:
                        st.write(result['answer'])

                    if 'visualization' in result and result['visualization']:
                        st.plotly_chart(result['visualization'], use_container_width=True)

                    if 'data' in result:
                        st.dataframe(result['data'], use_container_width=True)
                else:
                    st.warning("Could not understand the query. Please try rephrasing.")


def export_results(df):
    """Export analysis results."""
    st.subheader("📥 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export Data (CSV)"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="analysis_data.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("Export Summary (JSON)"):
            summary = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_types': df.dtypes.to_dict(),
                'insights': st.session_state.get('insights', {})
            }

            import json
            json_str = json.dumps(summary, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="analysis_summary.json",
                mime="application/json"
            )


def main():
    """Main application."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">📊 Victor AI Data Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("*Interactive data analysis with AI-powered insights*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        uploaded_file = st.file_uploader(
            "Upload Data File",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )

        if uploaded_file:
            st.session_state.data = load_data(uploaded_file)

        st.divider()

        if st.session_state.data is not None:
            st.write("📁 File Info")
            st.write(f"Name: {uploaded_file.name}")
            st.write(f"Size: {uploaded_file.size / 1024:.2f} KB")

    # Main content
    if st.session_state.data is not None:
        df = st.session_state.data

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Overview", "📊 Statistics", "📈 Visualizations", "🤖 Insights", "💬 Query"
        ])

        with tab1:
            display_data_overview(df)

        with tab2:
            display_statistical_analysis(df)

        with tab3:
            display_visualizations(df)

        with tab4:
            display_insights(df)

        with tab5:
            natural_language_query(df)

        # Export section
        st.divider()
        export_results(df)

    else:
        # Welcome message
        st.markdown("""
        ## 👋 Welcome to Victor AI Data Dashboard!

        **Features:**
        - 📊 Automated data profiling and statistics
        - 📈 Interactive visualizations
        - 🤖 AI-powered insights
        - 💬 Natural language querying
        - 📥 Export results

        **Getting Started:**
        1. Upload your data file using the sidebar
        2. Explore automatic analysis and insights
        3. Ask questions in natural language
        4. Export your findings

        **Supported Formats:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - JSON (.json)
        - Parquet (.parquet)

        Upload a file to get started!
        """)


if __name__ == "__main__":
    main()
