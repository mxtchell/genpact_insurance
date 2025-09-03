from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import jinja2
import pandas as pd
from ar_analytics import BreakoutAnalysis, BreakoutAnalysisTemplateParameterSetup, ArUtils
from ar_analytics.defaults import dimension_breakout_config, default_table_layout, get_table_layout_vars, \
    default_bridge_chart_viz, default_ppt_table_layout
from ar_analytics.helpers.df_meta_util import apply_metadata_to_layout_element
from skill_framework import SkillInput, SkillVisualization, skill, SkillParameter, SkillOutput, SuggestedQuestion, \
    ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from skill_framework.preview import preview_skill
from skill_framework.skills import ExportData
from genpact_formatting import apply_genpact_formatting_to_dataframe, genpact_format_number

logger = logging.getLogger(__name__)

@skill(
    name="Insurance Dimension Breakout",
    llm_name="insurance_dimension_breakout",
    description=dimension_breakout_config.description,
    capabilities=dimension_breakout_config.capabilities,
    limitations=dimension_breakout_config.limitations,
    example_questions=dimension_breakout_config.example_questions,
    parameter_guidance=dimension_breakout_config.parameter_guidance,
    parameters=[
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="If provided by the user, list time periods in a format 'q2 2023', '2021', 'jan 2023', 'mat nov 2022', 'mat q1 2021', 'ytd q4 2022', 'ytd 2023', 'ytd', 'mat', '<no_period_provided>' or '<since_launch>'. Use knowledge about today's date to handle relative periods and open ended periods. If given a range, for example 'last 3 quarters, 'between q3 2022 to q4 2023' etc, enumerate the range into a list of valid dates. Don't include natural language words or phrases, only valid dates like 'q3 2023', '2022', 'mar 2020', 'ytd sep 2021', 'mat q4 2021', 'ytd q1 2022', 'ytd 2021', 'ytd', 'mat', '<no_period_provided>' or '<since_launch>' etc."
        ),
        SkillParameter(
            name="metrics",
            is_multi=True,
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
            constrained_values=["Y/Y", "P/P", "None"],
            description="Growth type either Y/Y, P/P, or None"
        ),
        SkillParameter(
            name="other_filters",
            is_multi=True,
            constrained_to="filters"
        ),
        SkillParameter(
            name="growth_trend",
            constrained_to=None,
            constrained_values=["fastest growing", "highest growing", "highest declining", "fastest declining",
                                "smallest overall", "biggest overall"],
            description="indicates the trend type (fastest, highest, overall size) within a specified growth metric (year over year, period over period) for entities being analyzed."
        ),
        SkillParameter(
            name="calculated_metric_filters",
            description='This parameter allows filtering based on computed values like growth, delta, or share. The computed values are only available for metrics selected for this analysis. The available computations are growth, delta and share. It accepts a list of conditions, where each condition is a dictionary with:  metric: The metric being filtered. computation: The computation (growth, delta, share) operator: The comparison operator (">", "<", ">=", "<=", "between", "=="). value: The numeric threshold for filtering. If using "between", provide a list [min, max]. scale: the scale of value (percentage, bps, absolute)'
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt being used for max response.",
            default_value=dimension_breakout_config.max_prompt
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt being used for detailed insights.",
            default_value=dimension_breakout_config.insight_prompt
        ),
        SkillParameter(
            name="table_viz_layout",
            parameter_type="visualization",
            description="Table Viz Layout",
            default_value=default_table_layout
        ),
        SkillParameter(
            name="bridge_chart_viz_layout",
            parameter_type="visualization",
            description="Bridge Chart Viz Layout",
            default_value=default_bridge_chart_viz
        ),
        SkillParameter(
            name="table_ppt_layout",
            parameter_type="visualization",
            description="Table PPT Layout",
            default_value=default_ppt_table_layout
        )
    ]
)
def simple_breakout(parameters: SkillInput):
    param_dict = {"periods": [], "metrics": None, "limit_n": 10, "breakouts": None, "growth_type": None, "other_filters": [], "growth_trend": None, "calculated_metric_filters": None}
    print(f"Skill received following parameters: {parameters.arguments}")
    # Update param_dict with values from parameters.arguments if they exist
    for key in param_dict:
        if hasattr(parameters.arguments, key) and getattr(parameters.arguments, key) is not None:
            param_dict[key] = getattr(parameters.arguments, key)

    env = SimpleNamespace(**param_dict)
    BreakoutAnalysisTemplateParameterSetup(env=env)
    env.ba = BreakoutAnalysis.from_env(env=env)
    _ = env.ba.run_from_env()

    tables = env.ba.get_display_tables()
    param_info = [ParameterDisplayDescription(key=k, value=v) for k, v in env.ba.paramater_display_infomation.items()]

    insights_dfs = [env.ba.df_notes, env.ba.breakout_facts, env.ba.subject_facts]
    followups = env.ba.get_suggestions()

    viz, slides, insights, final_prompt, export_data = render_layout(tables,
                                                            env.ba.get_display_bridge_charts(),
                                                            env.ba.title,
                                                            env.ba.subtitle,
                                                            insights_dfs,
                                                            env.ba.warning_message,
                                                            env.ba.footnotes,
                                                            parameters.arguments.max_prompt,
                                                            parameters.arguments.insight_prompt,
                                                            parameters.arguments.table_viz_layout,
                                                            parameters.arguments.bridge_chart_viz_layout,
                                                            parameters.arguments.table_ppt_layout,
                                                            env)

    return SkillOutput(
        final_prompt=final_prompt,
        narrative=None,
        visualizations=viz,
        ppt_slides=slides,
        parameter_display_descriptions=param_info,
        followup_questions=[SuggestedQuestion(label=f.get("label"), question=f.get("question")) for f in followups if f.get("label")],
        export_data=[ExportData(name=name, id=df.max_metadata.get_id(), data=df) for name, df in export_data.items()]
    )

def find_footnote(footnotes, df):
    footnotes = footnotes or {}
    dim_note = None
    for col in df.columns:
        if col in footnotes:
            dim_note = footnotes.get(col)
            break
    return dim_note

def replace_period_placeholders(df, env):
    """Replace (CURRENT) and (PREVIOUS) in column names with actual period values"""
    if env is None or not hasattr(env, 'periods') or not env.periods:
        return df
    
    current_period = env.periods[0] if isinstance(env.periods, list) else env.periods
    # Capitalize quarters properly (q1 2025 -> Q1 2025)
    current_period = current_period.upper() if current_period.lower().startswith('q') else current_period
    
    # For previous period, handle both Y/Y and P/P
    previous_period = "PREVIOUS"
    if hasattr(env, 'growth_type') and env.growth_type in ["Y/Y", "P/P"]:
        if env.growth_type == "Y/Y":
            # Year over year - previous year
            if current_period.isdigit():
                previous_period = str(int(current_period) - 1)
            elif current_period.lower().startswith('q'):
                # For quarters like Q1 2025, extract year and subtract 1
                parts = current_period.split()
                if len(parts) == 2 and parts[1].isdigit():
                    prev_year = str(int(parts[1]) - 1)
                    previous_period = f"{parts[0]} {prev_year}"
                else:
                    previous_period = "PRIOR YEAR"
            else:
                previous_period = "PRIOR YEAR"
        elif env.growth_type == "P/P":
            # Period over period - previous period
            previous_period = "PRIOR PERIOD"
    
    # Replace column names - check all possible formats
    new_columns = []
    for col in df.columns:
        new_col = col
        
        # Check for various formats
        patterns_to_check = [
            ("(CURRENT)", f"({current_period})"),
            ("(PREVIOUS)", f"({previous_period})"),
            ("(Current)", f"({current_period})"),
            ("(Previous)", f"({previous_period})"),
            ("CURRENT", current_period),
            ("PREVIOUS", previous_period),
            ("Current", current_period),
            ("Previous", previous_period),
        ]
        
        for old_pattern, new_pattern in patterns_to_check:
            if old_pattern in new_col:
                new_col = new_col.replace(old_pattern, new_pattern)
        
        new_columns.append(new_col)
    
    df_copy = df.copy()
    df_copy.columns = new_columns
    return df_copy

def create_table_chart_layout(name, original_tables, general_vars, table_vars, table_id, viz_layout, env=None):
    """Create custom layout with table + bar chart + insights integrated together"""
    
    # Get the original table data for chart
    original_table = original_tables[name]
    
    # Find CURRENT period column for chart data - BULLETPROOF: NEVER use rank columns
    val_col = None
    
    print(f"DEBUG: All columns in table: {list(original_table.columns)}")
    print(f"DEBUG: Environment metrics: {getattr(env, 'metrics', None) if env else None}")
    print(f"DEBUG: Sample table data:")
    print(original_table.head(3))
    
    # Get all columns and filter out ANY rank-related columns completely
    all_columns = list(original_table.columns)
    non_rank_columns = []
    
    for col in all_columns:
        col_lower = col.lower()
        # BULLETPROOF: Skip ANY column that contains rank-related terms
        # Check exact match first, then substring
        if (col_lower == 'rank' or 
            any(term in col_lower for term in ['rank', 'top_rank', 'toprank', 'top ', 'position', 'order'])):
            print(f"DEBUG: REJECTING rank/position column: '{col}'")
            continue
        non_rank_columns.append(col)
    
    print(f"DEBUG: Non-rank columns available: {non_rank_columns}")
    
    # BULLETPROOF STRATEGY: Always use first METRIC column (skip dimension and utility columns)
    print("DEBUG: Finding first actual metric column...")
    
    for col in non_rank_columns:
        col_lower = col.lower()
        
        # Skip dimension column (usually first)
        if col == non_rank_columns[0]:
            print(f"DEBUG: SKIPPING dimension column: {col}")
            continue
            
        # Skip utility columns
        if col_lower in ['is_subject', 'subject']:
            print(f"DEBUG: SKIPPING utility column: {col}")
            continue
            
        # This must be a metric column - use it!
        print(f"DEBUG: FOUND first metric column: {col}")
        val_col = col
        break
    
    # Final fallback if somehow we have no good columns
    if not val_col and len(non_rank_columns) > 1:
        val_col = non_rank_columns[1]
        print(f"DEBUG: FALLBACK to second column: {val_col}")
    
    print(f"DEBUG: FINAL selected column: {val_col}")
    if val_col and not original_table.empty:
        print(f"DEBUG: Sample values from selected column: {original_table[val_col].head().tolist()}")
    
    # Extract metric name and time period 
    metric_name = "Value"
    time_period = "Current Period"
    
    if val_col:
        # Extract metric name (everything before first parenthesis or underscore)
        if '(' in val_col:
            metric_name = val_col.split('(')[0].strip()
        elif '_' in val_col:
            parts = val_col.split('_')
            metric_name = ' '.join(parts[:-1]).title() if len(parts) > 1 else val_col
        else:
            metric_name = val_col.replace('_', ' ').title()
    
    # Extract time period from env parameters - handle date ranges properly
    if env and hasattr(env, 'periods') and env.periods:
        periods = env.periods if isinstance(env.periods, list) else [env.periods]
        print(f"DEBUG: Processing {len(periods)} periods: {periods}")
        
        if len(periods) == 1:
            # Single period - capitalize quarters and years properly (q1 2025 -> Q1 2025)  
            period = periods[0]
            time_period = period.upper() if period.lower().startswith('q') else period
        else:
            # Multiple periods - create date range
            # Sort periods chronologically (not alphabetically)
            month_order = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            
            def sort_key(period):
                # Extract month and year for sorting
                parts = period.lower().split()
                if len(parts) >= 2:
                    month_part = parts[0][:3]  # First 3 chars of month
                    year_part = parts[1]
                    month_num = month_order.get(month_part, 0)
                    return (int(year_part), month_num)
                return (0, 0)
            
            sorted_periods = sorted(periods, key=sort_key)
            start_period = sorted_periods[0]
            end_period = sorted_periods[-1]
            print(f"DEBUG: Sorted periods from {start_period} to {end_period}")
            
            # Handle different period formats
            if start_period.lower().startswith('q') and end_period.lower().startswith('q'):
                # Quarter range: Q1 2025 to Q2 2025
                time_period = f"{start_period.upper()} to {end_period.upper()}"
            elif len(sorted_periods) >= 6:  # 6 or more months
                # For 6+ month ranges, show month range instead of listing all months
                # Extract year from periods to build proper range
                start_year = end_year = "2025"  # Default, but try to extract from periods
                if any(year in start_period for year in ['2024', '2025', '2026']):
                    for year in ['2024', '2025', '2026']:
                        if year in start_period:
                            start_year = year
                            break
                if any(year in end_period for year in ['2024', '2025', '2026']):
                    for year in ['2024', '2025', '2026']:
                        if year in end_period:
                            end_year = year
                            break
                
                # Map month abbreviations to full names
                month_map = {
                    'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April',
                    'may': 'May', 'jun': 'June', 'jul': 'July', 'aug': 'August', 
                    'sep': 'September', 'oct': 'October', 'nov': 'November', 'dec': 'December'
                }
                
                start_month = start_period.split()[0].lower()[:3]  # Get first 3 chars of month
                end_month = end_period.split()[0].lower()[:3]
                
                start_month_name = month_map.get(start_month, start_period.split()[0].title())
                end_month_name = month_map.get(end_month, end_period.split()[0].title())
                
                if start_year == end_year:
                    time_period = f"{start_month_name} to {end_month_name} {start_year}"
                else:
                    time_period = f"{start_month_name} {start_year} to {end_month_name} {end_year}"
            else:
                # Default: show range
                time_period = f"{start_period.title()} to {end_period.title()}"
        
        print(f"DEBUG: Final time_period for chart title: '{time_period}'")
    
    # Prepare chart data
    chart_data = []
    for idx, row in original_table.head(10).iterrows():
        country = str(row.iloc[0])
        value_str = str(row[val_col]) if pd.notna(row[val_col]) else "0"
        # Clean formatted value string (remove $, commas, etc.)
        value_clean = value_str.replace('$', '').replace(',', '').replace(' ', '')
        try:
            # Check if this is a percentage metric
            is_percentage = val_col and ('%' in val_col.lower() or 'percent' in val_col.lower())
            
            if is_percentage:
                # For percentages, use the original value as displayed (already formatted)
                raw_value = float(value_clean.replace('%', '')) if '%' in value_str else float(value_clean)
                formatted_value = value_str  # Keep original formatting like "12.52%"
            else:
                # For monetary values, use Genpact formatting with dollar sign
                raw_value = float(value_clean)
                formatted_value = f"${genpact_format_number(raw_value)}"
        except:
            raw_value = 0
            formatted_value = "0"
        chart_data.append({
            "name": country,
            "y": raw_value,
            "formatted": formatted_value
        })
    
    # Create bar chart configuration
    bar_chart = {
        "type": "highcharts",
        "config": {
            "chart": {"type": "column"},
            "title": {"text": f"{name} - {time_period}"},
            "xAxis": {"type": "category"},
            "yAxis": {
                "title": {"text": metric_name}
            },
            "tooltip": {
                "pointFormat": f"<b>{metric_name}: {{point.formatted}}</b>"
            },
            "series": [{
                "name": metric_name,
                "data": chart_data,
                "color": "#2E86C1"
            }],
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
    
    # Create custom layout combining table, chart, and insights with professional styling
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
                # Growth warning container
                {
                    "name": "CardContainer2",
                    "type": "CardContainer",
                    "children": "",
                    "minHeight": "40px",
                    "rows": 1,
                    "columns": 34,
                    "maxHeight": "40px",
                    "style": {
                        "borderRadius": "6.197px",
                        "background": "var(--Blue-50, #EFF6FF)",
                        "padding": "10px",
                        "paddingLeft": "20px",
                        "paddingRight": "20px"
                    },
                    "hidden": "{{hide_growth_warning}}"
                },
                # Warning text
                {
                    "name": "Header2",
                    "type": "Header",
                    "width": 32,
                    "children": "",
                    "text": "{{warning}}",
                    "style": {
                        "fontSize": "14px",
                        "fontWeight": "normal",
                        "textAlign": "left",
                        "verticalAlign": "start",
                        "color": "#1D4ED8",
                        "border": "null",
                        "textDecoration": "null",
                        "writingMode": "horizontal-tb",
                        "alignItems": "start",
                        "fontFamily": ""
                    },
                    "parentId": "CardContainer2",
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
                },
                # Insights card container
                {
                    "name": "CardContainer1",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "minHeight": "",
                    "maxHeight": "",
                    "style": {
                        "borderRadius": "11.911px",
                        "background": "var(--White, #FFF)",
                        "box-shadow": "0px 0px 8.785px 0px rgba(0, 0, 0, 0.10) inset",
                        "padding": "10px",
                        "fontFamily": "Arial"
                    },
                    "flexDirection": "row",
                    "hidden": False
                },
                # Insights header
                {
                    "name": "Header1",
                    "type": "Header",
                    "children": "",
                    "text": "Analysis Summary",
                    "style": {
                        "fontSize": "20px",
                        "fontWeight": "700",
                        "textAlign": "left",
                        "verticalAlign": "start",
                        "color": "#000000",
                        "backgroundColor": "#ffffff",
                        "border": "null",
                        "textDecoration": "null",
                        "writingMode": "horizontal-tb",
                        "borderBottom": "solid #DDD 2px"
                    },
                    "parentId": "CardContainer1",
                    "flex": "",
                    "hidden": False
                },
                # Insights content
                {
                    "name": "Markdown0",
                    "type": "Markdown",
                    "children": "",
                    "text": "{{exec_summary}}",
                    "style": {
                        "color": "#555",
                        "backgroundColor": "#ffffff",
                        "border": "null",
                        "fontSize": "15px"
                    },
                    "parentId": "CardContainer1",
                    "flex": "",
                    "hidden": False
                },
                # Footer
                {
                    "name": "Paragraph1",
                    "type": "Paragraph",
                    "children": "",
                    "text": "{{footer}}",
                    "style": {
                        "fontSize": "12px",
                        "fontWeight": "normal",
                        "textAlign": "left",
                        "verticalAlign": "start",
                        "color": "#000000",
                        "border": "null",
                        "textDecoration": "null",
                        "writingMode": "horizontal-tb"
                    },
                    "maxHeight": "32",
                    "hidden": "{{hide_footer}}"
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
            },
            {
                "name": "exec_summary",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "Markdown0",
                        "fieldName": "text"
                    }
                ]
            },
            {
                "name": "hide_growth_warning",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "CardContainer2",
                        "fieldName": "hidden"
                    }
                ]
            },
            {
                "name": "warning",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "Header2",
                        "fieldName": "text"
                    }
                ]
            },
            {
                "name": "footer",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "Paragraph1",
                        "fieldName": "text"
                    }
                ]
            },
            {
                "name": "hide_footer",
                "isRequired": False,
                "defaultValue": None,
                "targets": [
                    {
                        "elementName": "Paragraph1",
                        "fieldName": "hidden"
                    }
                ]
            }
        ]
    }
    
    return wire_layout(custom_layout, {**general_vars, **table_vars})

def render_layout(tables, bridge_chart_data, title, subtitle, insights_dfs, warnings, footnotes, max_prompt, insight_prompt, viz_layout, bridge_chart_viz_layout, table_ppt_layout, env=None):
    facts = []
    for i_df in insights_dfs:
        facts.append(i_df.to_dict(orient='records'))

    insight_template = jinja2.Template(insight_prompt).render(**{"facts": facts})
    max_response_prompt = jinja2.Template(max_prompt).render(**{"facts": facts})

    # adding insights - DISABLED FOR DEBUGGING
    # ar_utils = ArUtils()
    # insights = ar_utils.get_llm_response(insight_template)
    insights = "Debug mode - insights disabled"
    viz_list = []
    slides = []
    export_data = {}

    general_vars = {"headline": title if title else "Total",
					"sub_headline": subtitle or "Breakout Analysis",
					"hide_growth_warning": False if warnings else True,
					"exec_summary": insights if insights else "No Insights.",
					"warning": warnings}

    viz_layout = json.loads(viz_layout)

    # Store original tables for chart data before formatting
    original_tables = tables.copy()
    
    # Apply Genpact custom formatting to tables
    formatted_tables = {}
    for name, table in tables.items():
        if table is not None and not table.empty:
            # First replace period placeholders in column names
            table_with_periods = replace_period_placeholders(table, env)
            # Then apply Genpact formatting
            numeric_columns = table_with_periods.select_dtypes(include=['number']).columns.tolist()
            formatted_table = apply_genpact_formatting_to_dataframe(table_with_periods, numeric_columns)
            formatted_tables[name] = formatted_table
        else:
            formatted_tables[name] = table

    for name, table in formatted_tables.items():
        export_data[name] = table
        dim_note = find_footnote(footnotes, table)
        hide_footer = False if dim_note else True
        table_vars = get_table_layout_vars(table)
        table_vars["hide_footer"] = hide_footer
        table_vars["footer"] = f"*{dim_note.strip()}" if dim_note else "No additional info."
        
        # Create custom layout with table + chart + insights
        print(f"DEBUG: Creating chart layout for table '{name}'")
        custom_layout = create_table_chart_layout(name, original_tables, general_vars, table_vars, table.max_metadata.get_id(), viz_layout, env)
        print(f"DEBUG: Chart layout created successfully")
        viz_list.append(SkillVisualization(title=name, layout=custom_layout))
        
        if table_ppt_layout is not None:
            slide = wire_layout(json.loads(table_ppt_layout), {**general_vars, **table_vars})
            slides.append(slide)
        else:
            slides.append(custom_layout)


    print(f"DEBUG: Final viz_list count: {len(viz_list)}")
    print(f"DEBUG: Final viz titles: {[v.title for v in viz_list]}")
    return viz_list, slides, insights, max_response_prompt, export_data

if __name__ == '__main__':
    skill_input: SkillInput = simple_breakout.create_input(arguments={'metrics': ["opex_underwriting","nwp","underwriting_expense_ratio"], 'breakouts': ["line_of_business"], 'periods': ["jun 2025"], 'limit_n': "10"})
    out = simple_breakout(skill_input)
    preview_skill(simple_breakout, out)