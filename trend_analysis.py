from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import jinja2
from ar_analytics import AdvanceTrend, TrendTemplateParameterSetup, ArUtils
from ar_analytics.defaults import trend_analysis_config, default_trend_chart_layout, default_table_layout, \
    get_table_layout_vars, default_ppt_trend_chart_layout, default_ppt_table_layout
from skill_framework import SkillVisualization, skill, SkillParameter, SkillInput, SkillOutput, \
    ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from skill_framework.preview import preview_skill
from skill_framework.skills import ExportData

RUNNING_LOCALLY = False


logger = logging.getLogger(__name__)

@skill(
    name="Insurance Trend Analysis",
    llm_name=trend_analysis_config.llm_name,
    description=trend_analysis_config.description,
    capabilities=trend_analysis_config.capabilities,
    limitations=trend_analysis_config.limitations,
    example_questions=trend_analysis_config.example_questions,
    parameter_guidance=trend_analysis_config.parameter_guidance,
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
            name="time_granularity",
            is_multi=False,
            constrained_to="date_dimensions",
            description="time granularity provided by the user. only add if explicitly stated by user."
        ),
        SkillParameter(
            name="growth_type",
            constrained_to=None,
            constrained_values=["Y/Y", "P/P", "None"],
            description="Growth type either Y/Y, P/P, or None"
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt being used for max response.",
            default_value=trend_analysis_config.max_prompt
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt being used for detailed insights.",
            default_value=trend_analysis_config.insight_prompt
        ),
        SkillParameter(
            name="table_viz_layout",
            parameter_type="visualization",
            description="Table Viz Layout",
            default_value=default_table_layout
        ),
        SkillParameter(
            name="chart_viz_layout",
            parameter_type="visualization",
            description="Chart Viz Layout",
            default_value=default_trend_chart_layout
        ),
        SkillParameter(
            name="chart_ppt_layout",
            parameter_type="visualization",
            description="chart slide Viz Layout",
            default_value=default_ppt_trend_chart_layout
        ),
        SkillParameter(
            name="table_ppt_export_viz_layout",
            parameter_type="visualization",
            description="table slide Viz Layout",
            default_value=default_ppt_table_layout
        )
    ]
)
def trend(parameters: SkillInput):
    print(f"Skill received following parameters: {parameters.arguments}")
    param_dict = {"periods": [], "metrics": None, "limit_n": 10, "breakouts": [], "growth_type": None, "other_filters": [], "time_granularity": None}

    # Update param_dict with values from parameters.arguments if they exist
    for key in param_dict:
        if hasattr(parameters.arguments, key) and getattr(parameters.arguments, key) is not None:
            param_dict[key] = getattr(parameters.arguments, key)

    env = SimpleNamespace(**param_dict)
    TrendTemplateParameterSetup(env=env)
    env.trend = AdvanceTrend.from_env(env=env)
    df = env.trend.run_from_env()
    param_info = [ParameterDisplayDescription(key=k, value=v) for k, v in env.trend.paramater_display_infomation.items()]
    tables = [env.trend.display_dfs.get("Metrics Table")]

    insights_dfs = [env.trend.df_notes, env.trend.facts, env.trend.top_facts, env.trend.bottom_facts]

    charts = env.trend.get_dynamic_layout_chart_vars()
    
    # Filter out difference charts when growth_type is specified
    if param_dict.get("growth_type"):
        filtered_charts = {}
        for chart_name, chart_vars in charts.items():
            # Skip charts with "difference" in the name
            if "difference" in chart_name.lower() or "delta" in chart_name.lower():
                print(f"Skipping difference chart: {chart_name}")
                continue
            filtered_charts[chart_name] = chart_vars
        charts = filtered_charts

    viz, slides, insights, final_prompt = render_layout(charts,
                                                tables,
                                                env.trend.title,
                                                env.trend.subtitle,
                                                insights_dfs,
                                                env.trend.warning_message,
                                                parameters.arguments.max_prompt,
                                                parameters.arguments.insight_prompt,
                                                parameters.arguments.table_viz_layout,
                                                parameters.arguments.chart_viz_layout,
                                                parameters.arguments.chart_ppt_layout,
                                                parameters.arguments.table_ppt_export_viz_layout,
                                                param_dict)

    display_charts = env.trend.display_charts

    return SkillOutput(
        final_prompt=final_prompt,
        narrative=None,
        visualizations=viz,
        ppt_slides=slides,
        parameter_display_descriptions=param_info,
        followup_questions=[],
        export_data=[ExportData(name="Metrics Table", data=tables[0]),
                     *[ExportData(name=chart, data=display_charts[chart].get("df")) for chart in display_charts.keys()]]
    )

def map_chart_variables(chart_vars, prefix):
    """
    Maps prefixed chart variables to generic variable names expected by the layout.

    Args:
        chart_vars: Dictionary containing all chart variables with prefixes
        prefix: The prefix to extract (e.g., 'absolute_', 'growth_', 'difference_')

    Returns:
        Dictionary with mapped variables using generic names
    """
    suffixes = ['series', 'x_axis_categories', 'y_axis', 'metric_name', 'meta_df_id']

    mapped_vars = {}

    for suffix in suffixes:
        prefixed_key = f"{prefix}{suffix}"
        if prefixed_key in chart_vars:
            mapped_vars[suffix] = chart_vars[prefixed_key]

    if 'footer' in chart_vars:
        mapped_vars['footer'] = chart_vars['footer']
    if 'hide_footer' in chart_vars:
        mapped_vars['hide_footer'] = chart_vars['hide_footer']

    return mapped_vars

def render_layout(charts, tables, title, subtitle, insights_dfs, warnings, max_prompt, insight_prompt, table_viz_layout, chart_viz_layout, chart_ppt_layout, table_ppt_export_viz_layout, param_dict):
    facts = []
    for i_df in insights_dfs:
        facts.append(i_df.to_dict(orient='records'))

    insight_template = jinja2.Template(insight_prompt).render(**{"facts": facts})
    max_response_prompt = jinja2.Template(max_prompt).render(**{"facts": facts})

    # adding insights (disabled for testing)
    # ar_utils = ArUtils()
    # insights = ar_utils.get_llm_response(insight_template)
    insights = "Testing insights disabled"

    tab_vars = {"headline": title if title else "Total",
                "sub_headline": subtitle or "Trend Analysis",
                "hide_growth_warning": False if warnings else True,
                "exec_summary": insights if insights else "No Insight.",
                "warning": warnings}

    viz = []
    slides = []
    
    for name, chart_vars in charts.items():
        print(f"Processing chart: '{name}'")
        print(f"Chart vars keys: {list(chart_vars.keys())}")
        
        # Add hide_difference_chart logic when growth_type is specified
        if param_dict.get("growth_type"):
            chart_vars["hide_difference_chart"] = True
        
        # Filter out difference chart variables from chart_vars if hide_difference_chart is set
        filtered_chart_vars = {}
        for key, value in chart_vars.items():
            if key.startswith("difference_") and chart_vars.get("hide_difference_chart", False):
                print(f"  -> REMOVING difference variable: {key}")
                continue
            filtered_chart_vars[key] = value
        
        # If growth_type is specified, also remove difference from series data
        if param_dict.get("growth_type") and "series" in filtered_chart_vars:
            series = filtered_chart_vars["series"]
            if isinstance(series, list):
                # Filter out any series with "difference" in the name
                filtered_series = []
                for s in series:
                    if isinstance(s, dict) and "name" in s:
                        series_name = s["name"].lower()
                        if "difference" in series_name or "delta" in series_name:
                            print(f"  -> REMOVING difference series: {s['name']}")
                            continue
                    filtered_series.append(s)
                filtered_chart_vars["series"] = filtered_series
            
        filtered_chart_vars["footer"] = f"*{filtered_chart_vars['footer']}" if filtered_chart_vars.get('footer') else "No additional info."
        rendered = wire_layout(json.loads(chart_viz_layout), {**tab_vars, **filtered_chart_vars})
        viz.append(SkillVisualization(title=name, layout=rendered))

        prefixes = ["absolute_", "growth_", "difference_"]

        for prefix in prefixes:
            if (prefix in ["growth_"] and
                chart_vars.get("hide_growth_chart", False)):
                continue
            if (prefix in ["difference_"] and
                chart_vars.get("hide_difference_chart", False)):
                continue

            try:
                mapped_vars = map_chart_variables(chart_vars, prefix)
                slide = wire_layout(json.loads(chart_ppt_layout), {**tab_vars, **mapped_vars})
                slides.append(slide)
            except Exception as e:
                logger.error(f"Error rendering chart ppt slide for prefix '{prefix}' in chart '{name}': {e}")

    table_vars = get_table_layout_vars(tables[0])
    table = wire_layout(json.loads(table_viz_layout), {**tab_vars, **table_vars})
    viz.append(SkillVisualization(title="Metrics Table", layout=table))

    if table_ppt_export_viz_layout is not None:
        try: 
            table_slide = wire_layout(json.loads(table_ppt_export_viz_layout), {**tab_vars, **table_vars})
            slides.append(table_slide)
        except Exception as e:
            logger.error(f"Error rendering table ppt slide: {e}")
    else:
        slides.append(table)

    return viz, slides, insights, max_response_prompt

if __name__ == '__main__':
    skill_input: SkillInput = trend.create_input(arguments={
        'breakouts': ["distribution_channel"],
        'periods': ["q1 2024", "q2 2024", "q1 2025", "q2 2025"],
        'metrics': ["underwriting_profit"],
        'growth_type': "Y/Y",
        'time_granularity': "month"
    })
    out = trend(skill_input)
    preview_skill(trend, out)