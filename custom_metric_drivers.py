from __future__ import annotations
from types import SimpleNamespace

import pandas as pd
from skill_framework import SkillInput, SkillVisualization, skill, SkillParameter, SkillOutput, ParameterDisplayDescription
from skill_framework.preview import preview_skill
from skill_framework.skills import ExportData
from skill_framework.layouts import wire_layout

from ar_analytics import DriverAnalysis, DriverAnalysisTemplateParameterSetup, ArUtils
from ar_analytics.defaults import metric_driver_analysis_config, default_table_layout, get_table_layout_vars

import jinja2
import logging
import json

from genpact_formatting import genpact_format_number

logger = logging.getLogger(__name__)

@skill(
    name="Custom Metric Drivers",
    llm_name=metric_driver_analysis_config.llm_name,
    description=metric_driver_analysis_config.description,
    capabilities=metric_driver_analysis_config.capabilities,
    limitations=metric_driver_analysis_config.limitations,
    example_questions=metric_driver_analysis_config.example_questions,
    parameter_guidance=metric_driver_analysis_config.parameter_guidance,
    parameters=[
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="If provided by the user, list time periods in a format 'q2 2023', '2021', 'jan 2023', 'mat nov 2022', 'mat q1 2021', 'ytd q4 2022', 'ytd 2023', 'ytd', 'mat', '<no_period_provided>' or '<since_launch>'. Use knowledge about today's date to handle relative periods and open ended periods. If given a range, for example 'last 3 quarters, 'between q3 2022 to q4 2023' etc, enumerate the range into a list of valid dates. Don't include natural language words or phrases, only valid dates like 'q3 2023', '2022', 'mar 2020', 'ytd sep 2021', 'mat q4 2021', 'ytd q1 2022', 'ytd 2021', 'ytd', 'mat', '<no_period_provided>' or '<since_launch>' etc."
        ),
        SkillParameter(
            name="metric",
            is_multi=False,
            constrained_to="metrics"
        ),
        SkillParameter(
            name="limit_n",
            description="limit the number of values by this number",
            default_value=10
        ),
        SkillParameter(
            name="breakouts",
            is_multi=True,
            constrained_to="dimensions",
            description="breakout dimension(s) for analysis."
        ),
        SkillParameter(
            name="growth_type",
            constrained_to=None,
            constrained_values=["Y/Y", "P/P"],
            description="Growth type either Y/Y or P/P",
            default_value="Y/Y"
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True
        ),
        SkillParameter(
            name="calculated_metric_filters",
            description='This parameter allows filtering based on computed values like growth, delta, or share. The computed values are only available for metrics selected for this analysis. The available computations are growth, delta and share. It accepts a list of conditions, where each condition is a dictionary with:  metric: The metric being filtered. computation: The computation (growth, delta, share) operator: The comparison operator (">", "<", ">=", "<=", "between", "=="). value: The numeric threshold for filtering. If using "between", provide a list [min, max]. scale: the scale of value (percentage, bps, absolute)'
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt being used for max response.",
            default_value=metric_driver_analysis_config.max_prompt
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt being used for detailed insights.",
            default_value=metric_driver_analysis_config.insight_prompt
        ),
        SkillParameter(
            name="table_viz_layout",
            parameter_type="visualization",
            description="Table Viz Layout",
            default_value=default_table_layout
        )
    ]
)
def simple_metric_driver(parameters: SkillInput):
    param_dict = {"periods": [], "metric": "", "limit_n": 10, "breakouts": None, "growth_type": "Y/Y", "other_filters": [], "calculated_metric_filters": None}
    print(f"Skill received following parameters: {parameters.arguments}")
    # Update param_dict with values from parameters.arguments if they exist
    for key in param_dict:
        if hasattr(parameters.arguments, key) and getattr(parameters.arguments, key) is not None:
            param_dict[key] = getattr(parameters.arguments, key)

    env = SimpleNamespace(**param_dict)
    DriverAnalysisTemplateParameterSetup(env=env)
    env.da = DriverAnalysis.from_env(env=env)

    _ = env.da.run_from_env()

    results = env.da.get_display_tables()

    tables = {
        "Metrics": results['viz_metric_df']
    }
    tables.update(results['viz_breakout_dfs'])

    param_info = [ParameterDisplayDescription(key=k, value=v) for k, v in env.da.paramater_display_infomation.items()]

    insights_dfs = [env.da.df_notes, env.da.breakout_facts, env.da.subject_fact.get("df", pd.DataFrame())]

    warning_messages = env.da.get_warning_messages()

    viz, insights, final_prompt, export_data = render_layout(tables,
                                                            env.da.title,
                                                            env.da.subtitle,
                                                            insights_dfs,
                                                            warning_messages,
                                                            parameters.arguments.max_prompt,
                                                            parameters.arguments.insight_prompt,
                                                            parameters.arguments.table_viz_layout,
                                                            env)

    return SkillOutput(
        final_prompt=final_prompt,
        narrative=None,
        visualizations=viz,
        parameter_display_descriptions=param_info,
        followup_questions=[],
        export_data=[ExportData(name=name, data=df) for name, df in export_data.items()]
    )

def create_comparison_bar_chart(table, name, env=None):
    """Create horizontal bar chart comparing current vs previous period"""
    if table is None or table.empty:
        return None
    
    print(f"DEBUG HCHART: Creating horizontal chart for {name}")
    print(f"DEBUG HCHART: Table columns: {list(table.columns)}")
    print(f"DEBUG HCHART: Sample data:")
    print(table.head(3))
    
    # Find current and previous period columns - look for value columns
    current_col = None
    previous_col = None
    metric_col = None  # Store the metric name column
    
    for col in table.columns:
        col_lower = col.lower()
        # Skip first column (dimension name) and find metric columns
        if col == table.columns[0]:
            continue
        # Look for "Value" and "Prev Value" columns (common pattern in breakout tables)
        elif col_lower == 'value' and not current_col:
            current_col = col
            # Get metric name from environment or use generic "Value"
            if env and hasattr(env, 'metric'):
                # Convert metric name to readable format (e.g., total_opex -> Total Opex)
                metric_col = env.metric.replace('_', ' ').title()
            else:
                metric_col = "Value"
            print(f"DEBUG HCHART: Found current column: {col}, metric: {metric_col}")
        elif col_lower == 'prev value' or col_lower == 'previous value':
            previous_col = col
            print(f"DEBUG HCHART: Found previous column: {col}")
        # Look for columns with periods in parentheses
        elif '(' in col and ')' in col:
            period_str = col.split('(')[1].split(')')[0]
            if any(year in period_str for year in ['2025', '2026']):
                current_col = col
                metric_col = col.split('(')[0].strip()
                print(f"DEBUG HCHART: Found current column with period: {col}")
            elif any(year in period_str for year in ['2024', '2023']):
                previous_col = col
                print(f"DEBUG HCHART: Found previous column with period: {col}")
    
    if not current_col:
        print(f"DEBUG HCHART: No current column found, chart disabled")
        return None
    
    # Extract metric name and time periods
    metric_name = metric_col.split('(')[0].strip() if metric_col and '(' in metric_col else (metric_col if metric_col else "Value")
    
    # Get period labels from column names
    current_period = "Q2 2025"
    previous_period = "Q2 2024"
    
    # Try to extract periods from column names
    if '(' in current_col and ')' in current_col:
        current_period = current_col.split('(')[1].split(')')[0]
    elif env and hasattr(env, 'periods') and env.periods:
        period = env.periods[0] if isinstance(env.periods, list) else env.periods
        current_period = period.upper() if period.lower().startswith('q') else period
    
    if previous_col and '(' in previous_col and ')' in previous_col:
        previous_period = previous_col.split('(')[1].split(')')[0]
    elif env and hasattr(env, 'growth_type'):
        if env.growth_type == "Y/Y":
            # Year over year comparison
            if current_period.upper().startswith('Q') and len(current_period.split()) == 2:
                parts = current_period.split()
                prev_year = str(int(parts[1]) - 1)
                previous_period = f"{parts[0]} {prev_year}"
        elif env.growth_type == "P/P":
            # Period over period
            previous_period = "Q1 2025"  # Previous quarter
    
    # Prepare chart data - horizontal bars
    categories = []
    current_values = []
    previous_values = []
    
    for idx, row in table.head(10).iterrows():
        category = str(row.iloc[0])  # First column is dimension
        print(f"DEBUG HCHART: Processing {category}")
        
        # Get current period value
        current_val_str = str(row[current_col]) if pd.notna(row[current_col]) else "0"
        current_clean = current_val_str.replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
        try:
            current_val = float(current_clean)
        except:
            current_val = 0
        print(f"DEBUG HCHART: Current value: '{current_val_str}' -> {current_val}")
            
        # Get previous period value if column exists
        prev_val = 0
        if previous_col:
            prev_val_str = str(row[previous_col]) if pd.notna(row[previous_col]) else "0"
            prev_clean = prev_val_str.replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            try:
                prev_val = float(prev_clean)
            except:
                prev_val = 0
            print(f"DEBUG HCHART: Previous value: '{prev_val_str}' -> {prev_val}")
        
        categories.append(category)
        current_values.append(current_val)
        previous_values.append(prev_val)
    
    # Check if this is a percentage metric
    is_percentage = '%' in current_col.lower() or 'percent' in current_col.lower()
    
    # Format values for display
    current_formatted = []
    previous_formatted = []
    for curr, prev in zip(current_values, previous_values):
        if is_percentage:
            current_formatted.append(f"{curr:.1f}%")
            previous_formatted.append(f"{prev:.1f}%")
        else:
            current_formatted.append(f"${genpact_format_number(curr)}")
            previous_formatted.append(f"${genpact_format_number(prev)}")
    
    # Build series data
    series_data = [
        {
            "name": current_period,
            "data": [{"y": val, "formatted": fmt} for val, fmt in zip(current_values, current_formatted)],
            "color": "#2E86C1"
        }
    ]
    
    # Add previous period series only if we have previous data
    if previous_col:
        series_data.append({
            "name": previous_period,
            "data": [{"y": val, "formatted": fmt} for val, fmt in zip(previous_values, previous_formatted)],
            "color": "#8E44AD"
        })
    
    chart_title = f"{name} - {metric_name}"
    if previous_col:
        chart_title = f"{name} - {current_period} vs {previous_period}"
    
    print(f"DEBUG HCHART: Chart title: {chart_title}")
    print(f"DEBUG HCHART: Categories: {categories}")
    print(f"DEBUG HCHART: Series count: {len(series_data)}")
    
    # Create horizontal bar chart configuration
    bar_chart = {
        "type": "highcharts",
        "config": {
            "chart": {"type": "bar"},
            "title": {"text": chart_title},
            "xAxis": {"categories": categories},
            "yAxis": {
                "title": {"text": metric_name},
                "min": 0
            },
            "tooltip": {
                "pointFormat": "<b>{series.name}: {point.formatted}</b>",
                "valueSuffix": "%" if is_percentage else ""
            },
            "series": series_data,
            "plotOptions": {
                "bar": {
                    "dataLabels": {
                        "enabled": True,
                        "format": "{point.formatted}"
                    }
                }
            }
        }
    }
    
    return bar_chart

def create_vertical_metrics_chart(table, env=None):
    """Create vertical bar chart for metrics comparison on main tab"""
    if table is None or table.empty:
        return None
    
    print(f"DEBUG CHART: Table columns: {list(table.columns)}")
    print(f"DEBUG CHART: Table shape: {table.shape}")
    print(f"DEBUG CHART: Sample data:")
    print(table.head(3))
    
    # Find current and previous period columns - look for VALUE and PREV VALUE
    current_col = None
    previous_col = None
    
    for col in table.columns:
        col_lower = col.lower()
        if 'value' in col_lower and 'prev' not in col_lower:
            current_col = col
            print(f"DEBUG CHART: Found current column: {col}")
        elif 'prev value' in col_lower or ('previous' in col_lower and 'value' in col_lower):
            previous_col = col
            print(f"DEBUG CHART: Found previous column: {col}")
    
    if not current_col:
        print("DEBUG CHART: No current column found, chart disabled")
        return None
    
    if not previous_col:
        print("DEBUG CHART: No previous column found, will show single period chart")
    
    # Get period labels
    current_period = "Current"
    previous_period = "Previous"
    if env and hasattr(env, 'periods') and env.periods:
        period = env.periods[0] if isinstance(env.periods, list) else env.periods
        current_period = period.upper() if period.lower().startswith('q') else period
        # Calculate previous period
        if hasattr(env, 'growth_type') and env.growth_type == "Y/Y" and period.isdigit():
            previous_period = str(int(period) - 1)
        elif hasattr(env, 'growth_type') and env.growth_type == "Y/Y" and period.lower().startswith('q'):
            parts = period.split()
            if len(parts) == 2 and parts[1].isdigit():
                prev_year = str(int(parts[1]) - 1)
                previous_period = f"{parts[0].upper()} {prev_year}"
    
    # Prepare chart data - vertical columns
    categories = []
    current_values = []
    previous_values = []
    
    for idx, row in table.head(10).iterrows():
        metric_name = str(row.iloc[0])  # First column is metric name
        print(f"DEBUG CHART: Processing metric: {metric_name}")
        
        # Get current period value
        current_val_str = str(row[current_col]) if pd.notna(row[current_col]) else "0"
        current_clean = current_val_str.replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
        try:
            current_val = float(current_clean)
        except:
            current_val = 0
        print(f"DEBUG CHART: Current value: '{current_val_str}' -> {current_val}")
            
        # Get previous period value if column exists
        prev_val = 0
        if previous_col:
            prev_val_str = str(row[previous_col]) if pd.notna(row[previous_col]) else "0"
            prev_clean = prev_val_str.replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            try:
                prev_val = float(prev_clean)
            except:
                prev_val = 0
            print(f"DEBUG CHART: Previous value: '{prev_val_str}' -> {prev_val}")
        
        categories.append(metric_name)
        current_values.append(current_val)
        previous_values.append(prev_val)
    
    # Check if this is a percentage metric
    is_percentage = '%' in current_col.lower() or 'percent' in current_col.lower()
    
    # Format values for display
    current_formatted = []
    previous_formatted = []
    for curr, prev in zip(current_values, previous_values):
        if is_percentage:
            current_formatted.append(f"{curr:.1f}%")
            previous_formatted.append(f"{prev:.1f}%")
        else:
            current_formatted.append(genpact_format_number(curr, add_dollar_sign=True))
            previous_formatted.append(genpact_format_number(prev, add_dollar_sign=True))
    
    # Create chart series based on available data
    series_data = [
        {
            "name": current_period,
            "data": [{"y": val, "formatted": fmt} for val, fmt in zip(current_values, current_formatted)],
            "color": "#2E86C1"
        }
    ]
    
    # Add previous period series only if we have previous data
    if previous_col:
        series_data.append({
            "name": previous_period,
            "data": [{"y": val, "formatted": fmt} for val, fmt in zip(previous_values, previous_formatted)],
            "color": "#8E44AD"
        })
    
    chart_title = f"Metrics Overview - {current_period}"
    if previous_col:
        chart_title = f"Metrics Comparison - {current_period} vs {previous_period}"
    
    print(f"DEBUG CHART: Chart title: {chart_title}")
    print(f"DEBUG CHART: Categories: {categories}")
    print(f"DEBUG CHART: Series count: {len(series_data)}")
    
    # Create vertical column chart configuration
    column_chart = {
        "type": "highcharts",
        "config": {
            "chart": {"type": "column"},
            "title": {"text": chart_title},
            "xAxis": {"categories": categories},
            "yAxis": {
                "title": {"text": "Value"},
                "min": 0
            },
            "tooltip": {
                "pointFormat": "<b>{series.name}: {point.formatted}</b>",
                "valueSuffix": "%" if is_percentage else ""
            },
            "series": series_data,
            "plotOptions": {
                "column": {
                    "dataLabels": {
                        "enabled": True,
                        "format": "{point.formatted}"
                    }
                }
            }
        }
    }
    
    return column_chart

def create_table_chart_layout(name, table, general_vars, table_vars, viz_layout, env=None):
    """Create custom layout with table + horizontal bar chart integrated"""
    
    # Create the bar chart
    bar_chart = create_comparison_bar_chart(table, name, env)
    
    if not bar_chart:
        # Fallback to regular table layout if chart creation fails
        return wire_layout(json.loads(viz_layout), {**general_vars, **table_vars})
    
    # Create custom layout combining table and chart
    custom_layout = {
        "layoutJson": {
            "type": "Document",
            "rows": 90,
            "columns": 160,
            "rowHeight": "1.11%",
            "colWidth": "0.625%",
            "gap": "0px",
            "style": {
                "backgroundColor": "#ffffff",
                "width": "100%",
                "height": "max-content",
                "padding": "15px",
                "gap": "20px"
            },
            "children": [
                # Header card container
                {
                    "name": "CardContainer0",
                    "type": "CardContainer",
                    "children": "",
                    "minHeight": "80px",
                    "rows": 2,
                    "columns": 1,
                    "style": {
                        "border-radius": "11.911px",
                        "background": "#2563EB",
                        "padding": "10px",
                        "fontFamily": "Arial"
                    },
                    "hidden": False
                },
                # Title
                {
                    "name": "Header0",
                    "type": "Header",
                    "children": "",
                    "text": "{{headline}}",
                    "style": {
                        "fontSize": "20px",
                        "fontWeight": "700",
                        "color": "#ffffff",
                        "textAlign": "left",
                        "alignItems": "center"
                    },
                    "parentId": "CardContainer0",
                    "hidden": False
                },
                # Subtitle
                {
                    "name": "Paragraph0",
                    "type": "Paragraph",
                    "children": "",
                    "text": "{{sub_headline}}",
                    "style": {
                        "fontSize": "15px",
                        "fontWeight": "normal",
                        "textAlign": "center",
                        "verticalAlign": "start",
                        "color": "#fafafa",
                        "border": "null",
                        "textDecoration": "null",
                        "writingMode": "horizontal-tb",
                        "alignItems": "center"
                    },
                    "parentId": "CardContainer0",
                    "hidden": False
                },
                # Main content container
                {
                    "name": "FlexContainer4",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "250px",
                    "direction": "column",
                    "maxHeight": "1200px"
                },
                # Data table
                {
                    "name": "DataTable0",
                    "type": "DataTable",
                    "children": "",
                    "columns": [],
                    "data": [],
                    "parentId": "FlexContainer4",
                    "caption": "",
                    "styles": {
                        "td": {
                            "vertical-align": "middle"
                        }
                    }
                },
                # Chart container
                {
                    "name": "ChartContainer0",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "minHeight": "400px",
                    "style": {
                        "borderRadius": "11.911px",
                        "background": "var(--White, #FFF)",
                        "box-shadow": "0px 0px 8.785px 0px rgba(0, 0, 0, 0.10) inset",
                        "padding": "20px",
                        "margin": "20px 0",
                        "fontFamily": "Arial"
                    },
                    "parentId": "FlexContainer4"
                },
                # Chart
                {
                    "name": "HighchartsChart0",
                    "type": "HighchartsChart",
                    "children": "",
                    "style": {
                        "border": "none",
                        "borderRadius": "8px"
                    },
                    "options": bar_chart["config"],
                    "parentId": "ChartContainer0",
                    "flex": ""
                }
            ]
        },
        "inputVariables": [
            {
                "name": "col_defs",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "DataTable0",
                        "fieldName": "columns"
                    }
                ]
            },
            {
                "name": "data",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "DataTable0",
                        "fieldName": "data"
                    }
                ]
            },
            {
                "name": "headline",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "Header0",
                        "fieldName": "text"
                    }
                ]
            },
            {
                "name": "sub_headline",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "Paragraph0",
                        "fieldName": "text"
                    }
                ]
            }
        ]
    }
    
    return wire_layout(custom_layout, {**general_vars, **table_vars})

def create_metrics_tab_layout(table, general_vars, table_vars, viz_layout, env=None):
    """Create custom layout with vertical chart above table for main Metrics tab"""
    
    # Create the vertical chart
    vertical_chart = create_vertical_metrics_chart(table, env)
    
    if not vertical_chart:
        # Fallback to regular table layout if chart creation fails
        return wire_layout(json.loads(viz_layout), {**general_vars, **table_vars})
    
    # Create custom layout with chart above table
    custom_layout = {
        "layoutJson": {
            "type": "Document",
            "rows": 90,
            "columns": 160,
            "rowHeight": "1.11%",
            "colWidth": "0.625%",
            "gap": "0px",
            "style": {
                "backgroundColor": "#ffffff",
                "width": "100%",
                "height": "max-content",
                "padding": "15px",
                "gap": "20px"
            },
            "children": [
                # Header card container
                {
                    "name": "CardContainer0",
                    "type": "CardContainer",
                    "children": "",
                    "minHeight": "80px",
                    "rows": 2,
                    "columns": 1,
                    "style": {
                        "border-radius": "11.911px",
                        "background": "#2563EB",
                        "padding": "10px",
                        "fontFamily": "Arial"
                    },
                    "hidden": False
                },
                # Title
                {
                    "name": "Header0",
                    "type": "Header",
                    "children": "",
                    "text": "{{headline}}",
                    "style": {
                        "fontSize": "20px",
                        "fontWeight": "700",
                        "color": "#ffffff",
                        "textAlign": "left",
                        "alignItems": "center"
                    },
                    "parentId": "CardContainer0",
                    "hidden": False
                },
                # Subtitle
                {
                    "name": "Paragraph0",
                    "type": "Paragraph",
                    "children": "",
                    "text": "{{sub_headline}}",
                    "style": {
                        "fontSize": "15px",
                        "fontWeight": "normal",
                        "textAlign": "center",
                        "verticalAlign": "start",
                        "color": "#fafafa",
                        "border": "null",
                        "textDecoration": "null",
                        "writingMode": "horizontal-tb",
                        "alignItems": "center"
                    },
                    "parentId": "CardContainer0",
                    "hidden": False
                },
                # Main content container
                {
                    "name": "FlexContainer4",
                    "type": "FlexContainer",
                    "children": "",
                    "minHeight": "250px",
                    "direction": "column",
                    "maxHeight": "1200px"
                },
                # Chart container (above table)
                {
                    "name": "ChartContainer0",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "minHeight": "400px",
                    "style": {
                        "borderRadius": "11.911px",
                        "background": "var(--White, #FFF)",
                        "box-shadow": "0px 0px 8.785px 0px rgba(0, 0, 0, 0.10) inset",
                        "padding": "20px",
                        "margin": "20px 0",
                        "fontFamily": "Arial"
                    },
                    "parentId": "FlexContainer4"
                },
                # Chart
                {
                    "name": "HighchartsChart0",
                    "type": "HighchartsChart",
                    "children": "",
                    "style": {
                        "border": "none",
                        "borderRadius": "8px"
                    },
                    "options": vertical_chart["config"],
                    "parentId": "ChartContainer0",
                    "flex": ""
                },
                # Data table (below chart)
                {
                    "name": "DataTable0",
                    "type": "DataTable",
                    "children": "",
                    "columns": [],
                    "data": [],
                    "parentId": "FlexContainer4",
                    "caption": "",
                    "styles": {
                        "td": {
                            "vertical-align": "middle"
                        }
                    }
                }
            ]
        },
        "inputVariables": [
            {
                "name": "col_defs",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "DataTable0",
                        "fieldName": "columns"
                    }
                ]
            },
            {
                "name": "data",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "DataTable0",
                        "fieldName": "data"
                    }
                ]
            },
            {
                "name": "headline",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "Header0",
                        "fieldName": "text"
                    }
                ]
            },
            {
                "name": "sub_headline",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "Paragraph0",
                        "fieldName": "text"
                    }
                ]
            }
        ]
    }
    
    return wire_layout(custom_layout, {**general_vars, **table_vars})

def render_layout(tables, title, subtitle, insights_dfs, warnings, max_prompt, insight_prompt, viz_layout, env=None):
    facts = []
    for i_df in insights_dfs:
        facts.append(i_df.to_dict(orient='records'))

    insight_template = jinja2.Template(insight_prompt).render(**{"facts": facts})
    max_response_prompt = jinja2.Template(max_prompt).render(**{"facts": facts})

    # adding insights (temporarily disabled due to timeout issues)
    try:
        ar_utils = ArUtils()
        insights = ar_utils.get_llm_response(insight_template)
    except:
        insights = "Analysis insights"
    viz_list = []
    export_data = {}

    general_vars = {"headline": title if title else "Total",
                    "sub_headline": subtitle if subtitle else "Driver Analysis",
                    "hide_growth_warning": False if warnings else True,
                    "exec_summary": insights if insights else "No Insights.",
                    "warning": warnings}

    for name, table in tables.items():
        export_data[name] = table
        hide_footer = True
        table_vars = get_table_layout_vars(table, sparkline_col="sparkline")
        table_vars["hide_footer"] = hide_footer
        
        # Use custom layouts with charts
        if name == "Metrics":
            # Use vertical chart layout for main Metrics tab
            rendered = create_metrics_tab_layout(table, general_vars, table_vars, viz_layout, env)
        else:
            # Use horizontal chart layout for breakout tabs
            rendered = create_table_chart_layout(name, table, general_vars, table_vars, viz_layout, env)
        
        viz_list.append(SkillVisualization(title=name, layout=rendered))

    return viz_list, insights, max_response_prompt, export_data

if __name__ == '__main__':
    skill_input: SkillInput = simple_metric_driver.create_input(
        arguments={
  "breakouts": [
    "region"
  ],
  "metric": "underwriting_profit",
  "periods": [
    "q1 2025"
  ],
  "growth_type": "Y/Y"
})
    out = simple_metric_driver(skill_input)
    preview_skill(simple_metric_driver, out)