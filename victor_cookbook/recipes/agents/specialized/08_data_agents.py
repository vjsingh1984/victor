"""Specialized data analysis agent recipes.

These recipes focus on data tasks like exploration,
visualization, statistical analysis, and machine learning.
"""

RECIPE_CATEGORY = "agents/specialized"
RECIPE_DIFFICULTY = "intermediate"
RECIPE_TIME = "15 minutes"


async def data_exploration_agent(data_description: str):
    """Explore and summarize dataset."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["read", "python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Perform data exploration on:

{data_description}

Provide:
1. Data shape and structure
2. Data types and ranges
3. Missing values analysis
4. Summary statistics
5. Distribution analysis
6. Correlation analysis
7. Key findings and insights
8. Python code for exploration"""
    )

    return result.content


async def data_cleaning_agent(data_issues: str):
    """Generate data cleaning code."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Generate data cleaning code for:

{data_issues}

Provide:
1. Assessment of data quality issues
2. Cleaning strategy
3. Python code using pandas
4. Handling missing values
5. Outlier detection and treatment
6. Data type corrections
7. Validation checks"""
    )

    return result.content


async def feature_engineering_agent(
    dataset_description: str,
    target_variable: str
):
    """Design feature engineering approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.4
    )

    result = await agent.run(
        f"""Design feature engineering for:

DATASET: {dataset_description}
TARGET: {target_variable}

Provide:
1. Feature selection strategy
2. Feature creation ideas
3. Transformation techniques
4. Encoding approaches
5. Scaling/normalization
6. Dimensionality reduction
7. Python code examples"""
    )

    return result.content


async def statistical_analysis_agent(
    data_description: str,
    research_questions: list[str]
):
    """Design statistical analysis approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    questions_str = "\n".join(f"- {q}" for q in research_questions)

    result = await agent.run(
        f"""Design statistical analysis for:

DATA: {data_description}

QUESTIONS:
{questions_str}

Provide:
1. Appropriate statistical tests
2. Assumptions checking
4. Effect size calculations
5. Confidence intervals
6. Multiple testing correction
7. Interpretation guide
8. Python/R code"""
    )

    return result.content


async def visualization_agent(
    data_description: str,
    visualization_goals: list[str]
):
    """Generate data visualization code."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.4
    )

    goals_str = "\n".join(f"- {g}" for g in visualization_goals)

    result = await agent.run(
        f"""Generate visualization code for:

DATA: {data_description}

GOALS:
{goals_str}

Provide:
1. Recommended chart types
2. Python code using matplotlib/seaborn/plotly
3. Color schemes
4. Labels and annotations
5. Interactive features (if applicable)
6. Best practices for this data type"""
    )

    return result.content


async def time_series_analysis_agent(series_description: str):
    """Design time series analysis approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design time series analysis for:

{series_description}

Provide:
1. Decomposition approach
2. Trend analysis
3. Seasonality detection
4. Stationarity tests
5. Forecasting models (ARIMA, Prophet, etc.)
6. Evaluation metrics
7. Python code examples"""
    )

    return result.content


async def clustering_agent(data_description: str):
    """Design clustering analysis approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design clustering analysis for:

{data_description}

Provide:
1. Algorithm selection (K-means, DBSCAN, hierarchical)
2. Optimal cluster number determination
3. Feature selection/preprocessing
4. Validation approach (silhouette, Davies-Bouldin)
5. Cluster interpretation
6. Visualization approach
7. Python code examples"""
    )

    return result.content


async def classification_agent(
    data_description: str,
    target: str
):
    """Design classification model approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design classification model for:

DATA: {data_description}
TARGET: {target}

Provide:
1. Algorithm recommendations
2. Train/test split strategy
3. Cross-validation approach
4. Feature engineering
5. Hyperparameter tuning
6. Evaluation metrics
7. Handling class imbalance
8. Python code using scikit-learn"""
    )

    return result.content


async def regression_agent(
    data_description: str,
    target: str
):
    """Design regression model approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design regression model for:

DATA: {data_description}
TARGET: {target}

Provide:
1. Algorithm recommendations
2. Feature selection
3. Regularization approach
4. Residual analysis
5. Model interpretation
6. Evaluation metrics (RMSE, R², MAE)
7. Python code examples"""
    )

    return result.content


async def nlp_analysis_agent(text_data_description: str):
    """Design NLP analysis approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design NLP analysis for:

{text_data_description}

Provide:
1. Text preprocessing steps
2. Feature extraction (TF-IDF, embeddings, etc.)
3. Analysis approach (sentiment, topic modeling, NER)
4. Visualization recommendations
5. Model selection
6. Python code using spaCy/nltk/transformers
7. Evaluation approach"""
    )

    return result.content


async def anomaly_detection_agent(data_description: str):
    """Design anomaly detection approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design anomaly detection for:

{data_description}

Provide:
1. Algorithm selection (Isolation Forest, LOF, etc.)
2. Feature engineering for anomalies
3. Threshold determination
4. Evaluation strategy
5. Alerting approach
6. Visualization of anomalies
7. Python code examples"""
    )

    return result.content


async def ab_testing_agent(experiment_description: str):
    """Design A/B test analysis approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design A/B test analysis for:

{experiment_description}

Provide:
1. Hypothesis formulation
2. Sample size calculation
3. Randomization strategy
4. Metrics to track
5. Statistical tests
6. Significance level and power
7. Confidence intervals
8. Python code for analysis"""
    )

    return result.content


async def cohort_analysis_agent(data_description: str):
    """Design cohort analysis approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design cohort analysis for:

{data_description}

Provide:
1. Cohort definition strategy
2. Time period selection
3. Metrics to track
4. Retention calculation
5. Visualization approach (cohort matrix)
6. Insights extraction
7. Python code examples"""
    )

    return result.content


async def churn_prediction_agent(data_description: str):
    """Design customer churn prediction model."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design churn prediction model for:

{data_description}

Provide:
1. Feature engineering for churn
2. Target variable definition
3. Algorithm selection
4. Handling imbalanced data
5. Model evaluation (precision-recall, ROC)
6. Interpretability (SHAP, feature importance)
7. Deployment considerations
8. Python code examples"""
    )

    return result.content


async def recommendation_agent(data_description: str):
    """Design recommendation system approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design recommendation system for:

{data_description}

Provide:
1. Approach (collaborative filtering, content-based, hybrid)
2. Algorithm selection
3. Cold start strategy
4. Evaluation metrics (precision@k, recall@k)
5. Scalability considerations
6. Python code using surprise/lightfm
7. A/B testing approach"""
    )

    return result.content


async def forecasting_agent(series_description: str, forecast_horizon: int):
    """Design forecasting approach."""
    from victor import Agent

    agent = Agent.create(
        vertical="dataanalysis",
        tools=["python"],
        temperature=0.3
    )

    result = await agent.run(
        f"""Design forecasting model for:

{series_description}

HORIZON: {forecast_horizon} periods

Provide:
1. Exploratory analysis approach
2. Model selection (ARIMA, Prophet, LSTM, etc.)
3. Feature engineering for time series
4. Train/test split for time series
5. Evaluation metrics (MAPE, RMSE)
6. Prediction intervals
7. Python code examples"""
    )

    return result.content


async def demo_data_agents():
    """Demonstrate data agent recipes."""
    print("=== Data Analysis Agent Recipes ===\n")

    print("1. Data Exploration:")
    result = await data_exploration_agent(
        "CSV file with 10000 rows, columns: age, income, purchase_amount, date"
    )
    print(result[:300] + "...\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_data_agents())
