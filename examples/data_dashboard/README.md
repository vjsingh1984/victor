# Data Analysis Dashboard

An interactive data analysis dashboard powered by Victor AI, featuring automated insights, visualizations, and natural language querying capabilities.

## Features

- **Smart Data Loading**: Automatically detects file formats and schema
- **Automated Insights**: AI-powered analysis of your data
- **Interactive Visualizations**: Charts, graphs, and statistical plots
- **Natural Language Queries**: Ask questions about your data in plain English
- **Statistical Analysis**: Descriptive statistics, correlations, distributions
- **Anomaly Detection**: Automatic identification of outliers and anomalies
- **Data Cleaning**: Intelligent suggestions for data quality improvements
- **Export Options**: Save analysis results and charts

## Screenshots

The dashboard provides:
- Data overview with key statistics
- Interactive charts and visualizations
- Natural language query interface
- Automated insights panel

## Installation

```bash
# Navigate to demo directory
cd examples/data_dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Upload Data

1. Click "Browse files" in the sidebar
2. Select your data file (CSV, Excel, JSON, Parquet)
3. View automatic data analysis

### Ask Questions

Use the natural language query interface:

- "What is the average price?"
- "Show me the distribution of ages"
- "Find correlations between columns"
- "Identify outliers in the data"
- "Compare sales by region"

### Generate Reports

Click "Export Report" to download:
- Analysis summary (PDF)
- Charts (PNG)
- Data insights (JSON)

## Supported Data Formats

- **CSV**: Comma-separated values
- **Excel**: .xlsx and .xls files
- **JSON**: Nested and flat JSON
- **Parquet**: Apache Parquet format
- **Feather**: Efficient binary format

## Features in Detail

### 1. Automated Data Profiling

```python
from victor.dataanalysis import DataProfiler

profiler = DataProfiler()
profile = profiler.analyze(df)

# Shows:
# - Column types and statistics
# - Missing values
# - Unique values
# - Distributions
```

### 2. AI-Powered Insights

```python
from victor.dataanalysis import InsightGenerator

generator = InsightGenerator(orchestrator)
insights = generator.generate(df)

# Returns:
# - Key patterns
# - Trends
# - Anomalies
# - Recommendations
```

### 3. Natural Language Querying

```python
from victor.dataanalysis import NLQueryEngine

engine = NLQueryEngine(orchestrator)
result = engine.query("What is the average revenue by month?")

# Executes the appropriate pandas operation
# and returns the result with visualization
```

### 4. Statistical Analysis

```python
from victor.dataanalysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Descriptive statistics
stats = analyzer.describe(df)

# Correlation analysis
correlations = analyzer.correlate(df, method='pearson')

# Distribution analysis
distributions = analyzer.distributions(df)
```

### 5. Visualization

```python
from victor.dataanalysis import AutoVisualizer

visualizer = AutoVisualizer()

# Auto-select best chart type
charts = visualizer.suggest_charts(df)

# Generate specific visualizations
visualizer.histogram(df, 'column_name')
visualizer.scatter(df, 'x_col', 'y_col')
visualizer.heatmap(df.corr())
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           Streamlit Dashboard UI                     │
│         (app.py, pages/)                             │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          Data Analysis Engine                        │
│  - Data loading and cleaning                         │
│  - Statistical analysis                              │
│  - Visualization generation                          │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│    Data     │ │  Statistical│ │  AI-Powered │
│  Profiling  │ │   Analysis  │ │   Insights  │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌─────────────────┐
              │  Victor AI      │
              │  Orchestrator   │
              └─────────────────┘
```

## Example Workflows

### Sales Data Analysis

1. Upload sales CSV
2. View automatic insights:
   - Total revenue: $1.2M
   - Top product: Widget X (23% of sales)
   - Seasonal trend: +15% in Q4
3. Ask questions:
   - "What are the best performing regions?"
   - "Show sales trend over time"
   - "Compare product categories"
4. Export findings

### Customer Analytics

1. Upload customer data
2. Automatic segmentation
3. Churn prediction insights
4. Lifetime value analysis
5. Export customer segments

## Integration with Victor AI

This demo showcases the DataAnalysis vertical:

### Data Tools
- Data loading and parsing
- Cleaning and preprocessing
- Feature extraction
- Statistical analysis

### AI Capabilities
- Natural language query understanding
- Automated insight generation
- Anomaly detection
- Pattern recognition

### Visualization
- Auto chart selection
- Interactive plots
- Statistical charts
- Custom visualizations

## Performance Tips

- For large datasets (>1M rows), use Parquet or Feather format
- Enable caching in Streamlit for faster reloads
- Use data sampling for exploratory analysis
- Export charts instead of regenerating

## Testing

```bash
# Run tests
pytest tests/

# Test with sample data
python -m pytest tests/test_analysis.py -v
```

## Sample Data

The `sample_data/` directory includes example datasets:

- `sales_data.csv` - Sample sales transactions
- `customer_data.csv` - Customer information
- `stock_prices.csv` - Time series stock data

## Contributing

This is a demo for Victor AI. Contributions welcome!

## License

MIT License

## Support

- **Documentation**: https://victor-ai.readthedocs.io
- **Issues**: https://github.com/your-org/victor-ai/issues
