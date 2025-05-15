import pandas as pd

from core.constants import CLINICAL_VARIABLES_TO_DROP_TFR


def clean_clinical_data(clinical_data: pd.DataFrame) -> pd.DataFrame:

    clinical_data = clinical_data.drop(columns=CLINICAL_VARIABLES_TO_DROP_TFR)
    
    bcr_abl1_level_columns = [
        'BCR-ABL1 transcript level at 3 months (%)', 
        'BCR-ABL1 transcript level at 6 months (%)', 
        'BCR-ABL1 transcript level at 12 months (%)',
    ]
    for column in bcr_abl1_level_columns:
        clinical_data[column] = pd.to_numeric(clinical_data[column], errors='coerce')

    return clinical_data

def clean_wsi_data(wsi_image_data: pd.DataFrame) -> pd.DataFrame:

    # Convert proportion values (0-1) to percentages (0-100%)
    for column in wsi_image_data.columns:
        if column.split('_')[-1] == 'Proportion':
            wsi_image_data[column] = wsi_image_data[column]*100
    
    # Discard variables connected to periphery, center or lipids.
    # Periphery and sample results are discarded, because we have 
    # both wedge and squash samples and only the sample-level 
    # results matter. Lipids are not interesting for us.
    strings_to_check = ['Center', 'Periphery', 'Lipid']
    columns_to_drop = wsi_image_data.filter(regex='|'.join(strings_to_check)).columns
    wsi_image_data = wsi_image_data.drop(columns=columns_to_drop)
    
    # Discard other non-relevant variables
    wsi_image_data = wsi_image_data.drop(columns=[
        'Sample_Area',
        'Sample_Proportion', 
        'Infocus_Area', 
        'Out_of_focus_Area', 
        'Infocus_Proportion', 
        'Out_of_focus_Proportion', 
        'Megakaryocyte_Sample_Count_Proportion',
    ])
    return wsi_image_data


def remove_absolute_variables(df):
    
    # Find all columns in the DataFrame
    all_columns = df.columns

    # Create a list to hold columns to drop
    columns_to_drop = []

    # Loop through the columns to find absolute/percentage pairs
    for column in all_columns:
        # Check if the column name ends with '_Percentage'
        if column.endswith('_Percentage'):
            # Get the base name of the feature
            base_name = column.replace('_Percentage', '')
            # Check if the absolute count column exists
            if (base_name in all_columns) and (base_name != "Living cells"):
                # Add the absolute count column to the list of columns to drop
                columns_to_drop.append(base_name)

    # Drop the identified columns from the DataFrame
    df_cleaned = df.drop(columns=columns_to_drop)

    return df_cleaned


def clean_sample_statistic_data(df: pd.DataFrame) -> pd.DataFrame:

    # remove color statistics because not reliable in multicenter cohort, and mean variables because median is more robust
    df = df.drop(columns=df.filter(regex='red|green|blue|mean').columns)
    # remove roundness and convexity due to high collinearity with other more relevant shape variables
    df = df.drop(columns=df.filter(regex='roundness|convexity').columns)
    # Remove columns with more than 50% NaN values
    df = df.loc[:, df.isnull().mean() < 0.5]

    return df