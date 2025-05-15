from IPython.display import HTML
import pandas as pd
import re
import unicodedata


disable_code_blocks = HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')



def categorize_based_on_median(df, column_name, below_median=True, decimals=10):
    """
    Adds a new column to the DataFrame indicating whether values in the given column 
    are below the median, handling NaN values properly.

    Parameters:
    - df: DataFrame to modify
    - column_name: Name of the column to check
    - new_column_name: Name of the new column to create (optional)
    
    Returns:
    - Modified DataFrame with the new column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Calculate median excluding NaNs
    median_value = df[column_name].median(skipna=True)
    median_value = round(median_value, decimals)

    # Create the new column
    if below_median:
        df[f"{column_name} < median"] = df[column_name].apply(
            lambda x: x < median_value if pd.notna(x) else False
        )
    else:
        df[f"{column_name} > median"] = df[column_name].apply(
            lambda x: x > median_value if pd.notna(x) else False
        )
    
    return df, median_value


def format_p_value(p):
    if p < 0.001:
        return "< 0.001"
    elif p <= 0.01:
        return round(p, 3)
    else:
        return round(p, 2)


def transform_label(label, keep_cell_type=True):

    parts = label.split('_')
    
    # Map statistical terms to their readable equivalents
    stat_map = {'median': '(Mdn)', 'std': '(σ)'}
    
    # Process parts
    stat = stat_map.get(parts[-1], parts[-1])  # Replace median/std
    main_part = parts[-2].replace('-', ' ')  # Replace hyphens with spaces
    
    # Handle plurals and order
    if parts[0] == 'Eosinophils' and parts[1] in ['Mature', 'Immature']:
        cell_type = f"{parts[1]} Eosinophil"
    else:
        cell_type = parts[0][:-1] if parts[0].endswith('s') else parts[0]

    if keep_cell_type:
        return f"{cell_type} {main_part} {stat}"
    else:
        return f"{main_part.capitalize()} {stat}"


