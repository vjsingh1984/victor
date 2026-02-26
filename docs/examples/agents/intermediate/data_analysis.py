"""Data analysis examples."""

import asyncio
from victor import Agent


async def analyze_csv(file_path: str):
    """Analyze CSV file and provide insights."""
    agent = Agent.create(
        tools=["read", "python"],
        vertical="dataanalysis",
        temperature=0.3
    )
    result = await agent.run(
        f"""Analyze the CSV file at {file_path}.

        Provide:
        1. Column names and types
        2. Number of rows
        3. Summary statistics for numeric columns
        4. Unique values for categorical columns
        5. Any obvious patterns or outliers"""
    )
    return result.content


async def generate_plot_code(data_description: str, plot_type: str = "bar"):
    """Generate Python code to plot data."""
    agent = Agent.create(
        tools=["python"],
        vertical="dataanalysis",
        temperature=0.4
    )
    result = await agent.run(
        f"""Generate Python code using matplotlib to create a {plot_type} plot.

        Data: {data_description}

        Include:
        1. Import statements
        2. Data loading
        3. Plot configuration
        4. Labels and title
        5. Show the plot"""
    )
    return result.content


async def data_cleaning_suggestions(data_description: str):
    """Get suggestions for cleaning data."""
    agent = Agent.create(
        vertical="dataanalysis",
        temperature=0.4
    )
    result = await agent.run(
        f"""Suggest data cleaning steps for:
        {data_description}

        Consider:
        1. Missing values
        2. Outliers
        3. Data type conversions
        4. Duplicates
        5. Standardization"""
    )
    return result.content


async def statistical_analysis(file_path: str):
    """Perform statistical analysis on data."""
    agent = Agent.create(
        tools=["read", "python"],
        vertical="dataanalysis",
        temperature=0.3
    )
    result = await agent.run(
        f"""Perform statistical analysis on data in {file_path}.

        Include:
        1. Mean, median, mode
        2. Standard deviation
        3. Correlation analysis
        4. Distribution analysis
        5. Hypothesis test suggestions"""
    )
    return result.content


async def main():
    """Run data analysis examples."""
    print("=== Data Analysis Examples ===\n")

    # Example 1: Plot generation
    print("1. Generate Plot Code:")
    plot_code = await generate_plot_code(
        "Monthly sales data for 12 months",
        "line"
    )
    print(plot_code[:300] + "...\n")

    # Example 2: Cleaning suggestions
    print("2. Data Cleaning Suggestions:")
    suggestions = await data_cleaning_suggestions(
        "Customer data with missing emails and duplicate entries"
    )
    print(suggestions[:300] + "...\n")


if __name__ == "__main__":
    asyncio.run(main())
