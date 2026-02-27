"""Data processing workflow recipes.

These workflows handle data pipeline tasks like ETL,
cleaning, transformation, and analysis workflows.
"""

RECIPE_CATEGORY = "workflows/data_processing"
RECIPE_DIFFICULTY = "intermediate"
RECIPE_TIME = "25 minutes"


async def data_pipeline_workflow(
    source_path: str,
    destination_path: str,
    transformations: list[str]
):
    """End-to-end data processing pipeline."""
    import tempfile
    import os
    from victor import Agent

    transformations_str = "\n".join(f"- {t}" for t in transformations)

    workflow_yaml = f"""
name: "Data Processing Pipeline"
description: "ETL pipeline with validation and quality checks"

nodes:
  - id: "extract_data"
    type: "handler"
    config:
      tool: "read"
      arguments:
        file_path: "{{{{input.source_path}}}}"

  - id: "validate_schema"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Validate the schema of the data:
        {{{{extract_data.result}}}}

        Check:
        1. Data types are correct
        2. Required columns present
        3. No unexpected columns
        4. Value ranges are valid

        Report any schema issues.

  - id: "assess_quality"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Assess data quality:
        {{{{extract_data.result}}}}

        Analyze:
        1. Missing values per column
        2. Duplicate records
        3. Outliers
        4. Invalid values
        5. Consistency issues

        Generate quality report.

  - id: "clean_data"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Clean the data based on quality assessment:
        Data: {{{{extract_data.result}}}}
        Issues: {{{{assess_quality.output}}}}

        Apply:
        1. Handle missing values
        2. Remove duplicates
        3. Handle outliers
        4. Fix invalid values

        Provide cleaned data and cleaning log.

  - id: "transform_data"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Apply transformations to cleaned data:
        Data: {{{{clean_data.output}}}}

        Transformations to apply:
        {transformations_str}

        Provide transformation code and transformed data.

  - id: "validate_output"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Validate transformed data:
        {{{{transform_data.output}}}}

        Check:
        1. No null values introduced
        2. Data types correct
        3. Value ranges valid
        4. Record count reasonable
        5. No data corruption

        Generate validation report.

  - id: "save_data"
    type: "handler"
    config:
      tool: "write"
      arguments:
        file_path: "{{{{input.destination_path}}}}"
        content: "{{{{transform_data.output}}}}"

  - id: "generate_report"
    type: "agent"
    config:
      tools: ["write"]
      prompt: |
        Generate processing report:

        Source: {{{{input.source_path}}}}
        Destination: {{{{input.destination_path}}}}

        Schema validation: {{{{validate_schema.output}}}}
        Quality assessment: {{{{assess_quality.output}}}}
        Cleaning log: {{{{clean_data.output}}}}
        Transformations: {{{{transform_data.output}}}}
        Output validation: {{{{validate_output.output}}}}

        Include:
        1. Processing summary
        2. Records in/out
        3. Quality metrics
        4. Issues found and resolved
        5. Recommendations

        Save to: {{{{input.destination_path}}}}_report.md

edges:
  - from: "start"
    to: "extract_data"
  - from: "extract_data"
    to: "validate_schema"
  - from: "extract_data"
    to: "assess_quality"
  - from: "validate_schema"
    to: "clean_data"
  - from: "assess_quality"
    to: "clean_data"
  - from: "clean_data"
    to: "transform_data"
  - from: "transform_data"
    to: "validate_output"
  - from: "validate_output"
    to: "save_data"
  - from: "save_data"
    to: "generate_report"
  - from: "generate_report"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create(vertical="dataanalysis")
        result = await agent.run_workflow(
            workflow_path,
            input={
                "source_path": source_path,
                "destination_path": destination_path
            }
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def data_validation_workflow(
    data_path: str,
    validation_rules: list[str]
):
    """Data validation with custom rules workflow."""
    import tempfile
    import os
    from victor import Agent

    rules_str = "\n".join(f"- {r}" for r in validation_rules)

    workflow_yaml = f"""
name: "Data Validation"
description: "Validate data against custom business rules"

nodes:
  - id: "load_data"
    type: "handler"
    config:
      tool: "read"
      arguments:
        file_path: "{{{{input.data_path}}}}"

  - id: "parse_rules"
    type: "agent"
    config:
      prompt: |
        Parse and understand validation rules:
        {rules_str}

        For each rule:
        1. Identify the columns involved
        2. Define the validation logic
        3. Specify error conditions

  - id: "execute_validations"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Execute validation rules:
        Data: {{{{load_data.result}}}}
        Rules: {{{{parse_rules.output}}}}

        For each rule:
        1. Check all records
        2. Count violations
        3. Capture examples
        4. Calculate violation rate

        Provide detailed results.

  - id: "categorize_errors"
    type: "agent"
    config:
      prompt: |
        Categorize validation errors:
        {{{{execute_validations.output}}}}

        Group errors by:
        1. Severity (critical/high/medium/low)
        2. Type (format/range/logic/completeness)
        3. Frequency (common/rare)

        Generate error taxonomy.

  - id: "recommend_actions"
    type: "agent"
    config:
      prompt: |
        Based on validation results:
        {{{{execute_validations.output}}}}
        {{{{categorize_errors.output}}}}

        Recommend:
        1. Which errors to fix
        2. Prioritization order
        3. Cleaning approach
        4. Prevention strategies
        5. Monitoring recommendations

edges:
  - from: "start"
    to: "load_data"
  - from: "start"
    to: "parse_rules"
  - from: "load_data"
    to: "execute_validations"
  - from: "parse_rules"
    to: "execute_validations"
  - from: "execute_validations"
    to: "categorize_errors"
  - from: "categorize_errors"
    to: "recommend_actions"
  - from: "recommend_actions"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create(vertical="dataanalysis")
        result = await agent.run_workflow(
            workflow_path,
            input={"data_path": data_path}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def data_enrichment_workflow(
    base_data_path: str,
    enrichment_sources: list[str]
):
    """Enrich data with external sources."""
    import tempfile
    import os
    from victor import Agent

    sources_str = "\n".join(f"- {s}" for s in enrichment_sources)

    workflow_yaml = f"""
name: "Data Enrichment"
description: "Join and enrich base data with external sources"

nodes:
  - id: "load_base_data"
    type: "handler"
    config:
      tool: "read"
      arguments:
        file_path: "{{{{input.base_data_path}}}}"

  - id: "analyze_base_data"
    type: "agent"
    config:
      prompt: |
        Analyze base data structure:
        {{{{load_base_data.result}}}}

        Identify:
        1. Primary key columns
        2. Foreign key opportunities
        3. Missing attributes
        4. Enrichment opportunities

  - id: "design_enrichment"
    type: "agent"
    config:
      prompt: |
        Design enrichment strategy:
        Base data: {{{{analyze_base_data.output}}}}

        Available sources:
        {sources_str}

        Plan:
        1. Which sources to join
        2. Join keys
        3. Expected new attributes
        4. Join type (left/inner/full)

  - id: "perform_enrichment"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Perform data enrichment:
        Base: {{{{load_base_data.result}}}}
        Strategy: {{{{design_enrichment.output}}}}

        Execute the enrichment plan.
        Handle:
        - Missing keys
        - Duplicate matches
        - Data type mismatches
        - Conflicting values

        Provide enriched dataset.

  - id: "verify_enrichment"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Verify enrichment results:
        {{{{perform_enrichment.output}}}}

        Check:
        1. Expected new columns present
        2. Join success rate
        3. Data quality maintained
        4. No unexpected nulls introduced

        Generate verification report.

edges:
  - from: "start"
    to: "load_base_data"
  - from: "load_base_data"
    to: "analyze_base_data"
  - from: "analyze_base_data"
    to: "design_enrichment"
  - from: "design_enrichment"
    to: "perform_enrichment"
  - from: "perform_enrichment"
    to: "verify_enrichment"
  - from: "verify_enrichment"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create(vertical="dataanalysis")
        result = await agent.run_workflow(
            workflow_path,
            input={"base_data_path": base_data_path}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def data_aggregation_workflow(
    data_path: str,
    group_by_columns: list[str],
    aggregations: list[str]
):
    """Data aggregation and summarization workflow."""
    import tempfile
    import os
    from victor import Agent

    group_by_str = ", ".join(group_by_columns)
    agg_str = "\n".join(f"- {a}" for a in aggregations)

    workflow_yaml = f"""
name: "Data Aggregation"
description: "Group and aggregate data for analysis"

nodes:
  - id: "load_data"
    type: "handler"
    config:
      tool: "read"
      arguments:
        file_path: "{{{{input.data_path}}}}"

  - id: "analyze_distribution"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Analyze data distribution:
        {{{{load_data.result}}}}

        For grouping columns: {group_by_str}

        Provide:
        1. Unique values per group column
        2. Cardinality analysis
        3. Distribution visualizations
        4. Skewness assessment

  - id: "design_aggregation"
    type: "agent"
    config:
      prompt: |
        Design aggregation strategy:
        Group by: {group_by_str}
        Aggregations: {agg_str}

        Consider:
        1. Appropriate aggregation functions
        2. Handling nulls in groups
        3. Handling nulls in measures
        4. Multi-level aggregation if needed

  - id: "perform_aggregation"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Perform aggregation:
        Data: {{{{load_data.result}}}}
        Strategy: {{{{design_aggregation.output}}}}

        Execute aggregations using pandas.
        Provide:
        1. Aggregated data
        2. Row counts before/after
        3. Summary statistics

  - id: "validate_results"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Validate aggregation results:
        {{{{perform_aggregation.output}}}}

        Verify:
        1. All groups represented
        2. Aggregation logic correct
        3. No data loss (unexpectedly)
        4. Totals match expectations

edges:
  - from: "start"
    to: "load_data"
  - from: "load_data"
    to: "analyze_distribution"
  - from: "analyze_distribution"
    to: "design_aggregation"
  - from: "design_aggregation"
    to: "perform_aggregation"
  - from: "perform_aggregation"
    to: "validate_results"
  - from: "validate_results"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create(vertical="dataanalysis")
        result = await agent.run_workflow(
            workflow_path,
            input={"data_path": data_path}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def data_merging_workflow(
    primary_path: str,
    secondary_paths: list[str],
    merge_strategy: str
):
    """Merge multiple datasets together."""
    import tempfile
    import os
    from victor import Agent

    secondary_str = "\n".join(f"- {s}" for s in secondary_paths)

    workflow_yaml = f"""
name: "Data Merging"
description: "Merge multiple datasets into unified view"

nodes:
  - id: "load_primary"
    type: "handler"
    config:
      tool: "read"
      arguments:
        file_path: "{{{{input.primary_path}}}}"

  - id: "load_secondary"
    type: "agent"
    config:
      tools: ["read"]
      prompt: |
        Load all secondary datasets:
        {secondary_str}

        For each:
        1. Load the data
        2. Identify schema
        3. Identify potential merge keys

  - id: "analyze_schemas"
    type: "agent"
    config:
      prompt: |
        Analyze schemas for compatibility:
        Primary: {{{{load_primary.result}}}}
        Secondary: {{{{load_secondary.output}}}}

        Identify:
        1. Common columns
        2. Conflicting columns (same name, different meaning)
        3. Type mismatches
        4. Merge key candidates

  - id: "design_merge"
    type: "agent"
    config:
      prompt: |
        Design merge strategy:
        Strategy type: {{{{input.merge_strategy}}}}
        Analysis: {{{{analyze_schemas.output}}}}

        Specify:
        1. Merge keys
        2. Join types (left/right/inner/full)
        3. Column conflict resolution
        4. Handling of unmatched records

  - id: "execute_merge"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Execute the merge:
        Primary: {{{{load_primary.result}}}}
        Secondary: {{{{load_secondary.output}}}}
        Design: {{{{design_merge.output}}}}

        Perform the merge using pandas.
        Handle:
        - Duplicate column names (suffixes)
        - Unmatched records per strategy
        - Data type alignment

        Provide merged dataset and merge statistics.

  - id: "validate_merge"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Validate merge results:
        {{{{execute_merge.output}}}}

        Check:
        1. Expected row count
        2. No unexpected duplicates
        3. Key uniqueness maintained
        4. Data integrity preserved
        5. Null value patterns reasonable

edges:
  - from: "start"
    to: "load_primary"
  - from: "start"
    to: "load_secondary"
  - from: "load_primary"
    to: "analyze_schemas"
  - from: "load_secondary"
    to: "analyze_schemas"
  - from: "analyze_schemas"
    to: "design_merge"
  - from: "design_merge"
    to: "execute_merge"
  - from: "execute_merge"
    to: "validate_merge"
  - from: "validate_merge"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create(vertical="dataanalysis")
        result = await agent.run_workflow(
            workflow_path,
            input={
                "primary_path": primary_path,
                "merge_strategy": merge_strategy
            }
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def demo_data_workflows():
    """Demonstrate data processing workflows."""
    print("=== Data Processing Workflow Recipes ===\n")

    print("1. Data Pipeline:")
    result = await data_pipeline_workflow(
        "data/input.csv",
        "data/output.csv",
        ["Normalize text columns", "Convert dates", "Calculate derived fields"]
    )
    print(result[:400] + "...\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_data_workflows())
