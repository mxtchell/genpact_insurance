# Fixed chart creation logic that avoids rank columns

def create_table_chart_layout(name, original_tables, general_vars, table_vars, table_id, viz_layout, env=None):
    """Create custom layout with table + bar chart + insights integrated together"""
    
    # Get the original table data for chart
    original_table = original_tables[name]
    
    # Find the actual metric column (not rank column) 
    val_col = None
    
    # First, try to find columns that match requested metrics
    if env and hasattr(env, 'metrics') and env.metrics:
        for metric in env.metrics:
            metric_lower = metric.lower().replace('_', ' ')
            for col in original_table.columns:
                col_lower = col.lower()
                # Skip rank columns completely
                if 'rank' in col_lower or 'top ' in col_lower:
                    continue
                # Look for metric name in column
                if metric_lower in col_lower and ('current' in col_lower or not any(x in col_lower for x in ['previous', 'change', '%', 'growth'])):
                    val_col = col
                    break
            if val_col:
                break
    
    # Fallback: find any non-rank current period column
    if not val_col:
        for col in original_table.columns:
            col_lower = col.lower()
            # Skip rank columns completely
            if 'rank' in col_lower or 'top ' in col_lower:
                continue
            # Look for current period column
            if 'current' in col_lower or ('(' in col and any(year in col for year in ['2024', '2025'])):
                val_col = col
                break
    
    if not val_col:
        # No valid column found, return regular table
        return wire_layout(json.loads(viz_layout), {**general_vars, **table_vars})
    
    # Rest of the chart creation logic stays the same...
    # [Include the rest of your existing chart logic here]