"""Module for creating features from the input data variables"""

import pandas as pd
import numpy as np


def calculate_sample_cellularity(wsi_image_data: pd.DataFrame) -> pd.DataFrame:
    # create cellularity variables
    wsi_image_data["Cellularity"] = np.log((wsi_image_data["Eukaryote_Sample_Proportion"]/100 + wsi_image_data["Stroma_Sample_Proportion"]/100) * wsi_image_data["Eukaryote_Sample_Proportion"]/100)
    return wsi_image_data

def categorize_variable(series, threshold1, threshold2=None, comparison_type='less'):
    """
    Categorizes a pandas Series based on one or two threshold values.
    
    Parameters:
    - series: pandas Series containing the continuous variable.
    - threshold1: First threshold value.
    - threshold2: Second threshold value (optional).
    - comparison_type: 'less' or 'greater' to specify comparison for single threshold.
    
    Returns:
    - A pandas Series with categories 'yes', 'no', or NaN.
    """
    if threshold2 is None:
        # Single threshold case
        if comparison_type == 'less':
            return np.where(pd.isna(series), pd.NA,
                            np.where(series < threshold1, 'yes', 'no'))
        elif comparison_type == 'greater':
            return np.where(pd.isna(series), pd.NA,
                            np.where(series > threshold1, 'yes', 'no'))
        else:
            raise ValueError("comparison_type must be either 'less' or 'greater'")
    else:
        # Two thresholds case
        return np.where(pd.isna(series), pd.NA,
                        np.where((series < threshold1) & (series > threshold2), 'yes', 'no'))

    
def create_more_clinical_features(df):
  
    # Patient age
    df['Patient age at diagnosis_categorized'] = pd.cut(
        df['Patient age at diagnosis (completed years)'], 
        bins=[-float('inf'), 40, 50, 60, float('inf')], 
        labels=['0-40', '41-50', '51-60', '61-inf'], 
        right=True,
    )
    
    df['EUTOS score-risk class_low'] = np.where(pd.isna(df['EUTOS score']), pd.NA, np.where(df['EUTOS score'] <= 87, 'yes', 'no'))
    
    # patient age < 50
    df['Patient age at diagnosis > 50'] = np.where(
        pd.isna(df['Patient age at diagnosis (completed years)']), 
        pd.NA, 
        np.where(df['Patient age at diagnosis (completed years)'] > 50, 'yes', 'no'))
    
    # spleen size is larger than 0 cm
    df['Spleen size at diagnosis (max. distance from costal margin) > 0'] = np.where(
        pd.isna(df['Spleen size at diagnosis (max. distance from costal margin)']), 
        pd.NA, 
        np.where(df['Spleen size at diagnosis (max. distance from costal margin)'] > 0, 'yes', 'no'))
    
    
    # Hasford score
    df['Hasford score-risk class'] = pd.cut(
        df['Hasford score'], 
        bins=[-float('inf'), 780, 1481, float('inf')], 
        labels=['low', 'intermediate', 'high'], 
        right=True,
    )

    # Sokal score
    df['Sokal score-risk class'] = pd.cut(
        df['Sokal score'], 
        bins=[-float('inf'), 0.8, 1.2, float('inf')], 
        labels=['low', 'intermediate', 'high'], 
        right=False,
    )

    # EUTOS score
    df['EUTOS score-risk class_low'] = np.where(pd.isna(df['EUTOS score']), pd.NA, np.where(df['EUTOS score'] <= 87, 'yes', 'no'))

    # ELTS score
    df['ELTS score-risk class'] = pd.cut(
        df['ELTS score'], 
        bins=[-float('inf'), 1.5680, 2.2185, float('inf')], 
        labels=['low', 'intermediate', 'high'], 
        right=True,
    )

    # Early relapse
    df['Relapse within 6 months'] = np.select(
        condlist=[
            (df['Timepoint of possible relapse after first TKI discontinuation (months)'] <= 6) & (df['Relapse after first TKI discontinuation'] == 'yes'),
            (df['Timepoint of possible relapse after first TKI discontinuation (months)'] > 6) & (df['Relapse after first TKI discontinuation'] == 'yes'),
            (df['Relapse after first TKI discontinuation'] == 'no'),
        ],
        choicelist=['yes', 'no', pd.NA],
        default=pd.NA
    )

    # MMR at 12 months
    df['MMR at 12 months'] = np.select(
        condlist=[
            (df['Time to MMR (months)'] <= 12),
            (df['Time to MMR (months)'] > 12)
        ],
        choicelist=['yes', 'no'],
        default=pd.NA
    )
    
    # MMR at 6 months
    df['MMR at 6 months'] = np.select(
        condlist=[
            (df['Time to MMR (months)'] <= 6),
            (df['Time to MMR (months)'] > 6)
        ],
        choicelist=['yes', 'no'],
        default=pd.NA
    )
    
    # MR4.0 at 12 months
    df['MR4.0 at 12 months'] = np.select(
        condlist=[
            (df['Time to MR4.0 (months)'] <= 12),
            (df['Time to MR4.0 (months)'] > 12)
        ],
        choicelist=['yes', 'no'],
        default=pd.NA
    )
    
    df['Time to TKI discontinuation < 6 years']  = np.select(
        condlist=[
            (df['Time to TKI discontinuation (months)'] < 72),
            (df['Time to TKI discontinuation (months)'] >= 72)
        ],
        choicelist=['yes', 'no'],
        default=pd.NA
    )
    return df


def create_more_cell_count_features(df):

    df['Mature granulopoietic cells'] = df[['Basophils', 'Eosinophils', 'Neutrophils']].sum(axis=1)
    df['Immature granulopoietic cells'] = df[['Metamyelocytes', 'Myelocytes', 'Promyelocytes']].sum(axis=1)
    if "Granulopoietic cells" not in df.columns:
        df['Granulopoietic cells'] = df[['Immature granulopoietic cells', 'Mature granulopoietic cells']].sum(axis=1)
        df['Granulopoietic cells_Percentage'] = 100 * df['Granulopoietic cells'] / df['Living cells']
    if "Erythropoietic cells" not in df.columns:
        df['Erythropoietic cells'] = df[['Proerythroblasts', 'Erythroblasts']].sum(axis=1)
        df['Erythropoietic cells_Percentage'] = 100 * df['Erythropoietic cells'] / df['Living cells']

    df['Mature granulopoietic cells_Percentage'] = 100 * df['Mature granulopoietic cells'] / df['Living cells']
    df['Immature granulopoietic cells_Percentage'] = 100 * df['Immature granulopoietic cells'] / df['Living cells']
    df['Neutrophils-Eosinophils_Percentage'] = 100 * df[['Eosinophils', 'Neutrophils']].sum(axis=1) / df['Living cells']

    df['Neutrophils-to-Basophils Eosinophils_Ratio'] = df['Neutrophils'] / df[['Basophils', 'Eosinophils']].sum(axis=1)
    df['Mature granulocytes-to-All granulocytes_Ratio'] = df['Mature granulopoietic cells'] / df['Granulopoietic cells']
    df['Immature granulopoietic cells-to-All granulopoietic cells_Ratio'] = df['Immature granulopoietic cells'] / df['Granulopoietic cells']
    df['Mature granulopoietic cells-to-Immature granulopoietic cells_Ratio'] = df['Mature granulopoietic cells'] / df['Immature granulopoietic cells']
    df['Myelocytes+Metamyelocytes_Percentage'] = (df[['Myelocytes', 'Metamyelocytes']].sum(axis=1)) / df['Living cells']

    df['ME-ratio'] = (df['Granulopoietic cells'] + df['Monocytes'] + df['Promonocytes']) / df['Erythropoietic cells'] 
    
    df['Megakaryocytes-to-All granulopoietic cells_Ratio'] = df['Megakaryocytes'] / df['Granulopoietic cells']
    df['Monocytes-to-All-Granulopoietic cells_Ratio'] = df['Monocytes'] / df['Granulopoietic cells']
    
    df['Neutrophils_to_Mature granulopoietic cells_Ratio'] = df['Neutrophils'] / df['Mature granulopoietic cells']
    df['Basophils_to_Mature granulopoietic cells_Ratio'] = df['Basophils'] / df['Mature granulopoietic cells']
    df['Eosinophils_to_Mature granulopoietic cells_Ratio'] = df['Eosinophils'] / df['Mature granulopoietic cells']
    if 'Eosinophils_Mature' in df.columns:
        df['Eosinophils_Mature_to_Mature granulopoietic cells_Ratio'] = df['Eosinophils_Mature'] / df['Mature granulopoietic cells']
        df['Eosinophils_Mature_to_Granulopoietic cells_Ratio'] = df['Eosinophils_Mature'] / df['Granulopoietic cells']
    if 'Eosinophils_Immature' in df.columns:
        df['Eosinophils_Immature_to_Mature granulopoietic cells_Ratio'] = df['Eosinophils_Immature'] / df['Mature granulopoietic cells']
        df['Eosinophils_Immature_to_Granulopoietic cells_Ratio'] = df['Eosinophils_Immature'] / df['Granulopoietic cells']
    
    df['Neutrophils_to_Granulopoietic cells_Ratio'] = df['Neutrophils'] / df['Granulopoietic cells']
    df['Basophils_to_Granulopoietic cells_Ratio'] = df['Basophils'] / df['Granulopoietic cells']
    df['Eosinophils_to_Granulopoietic cells_Ratio'] = df['Eosinophils'] / df['Granulopoietic cells']
    
    df['Metamyelocytes_to_Granulopoietic cells_Ratio'] = df['Metamyelocytes'] / df['Granulopoietic cells']
    df['Myelocytes_to_Granulopoietic cells_Ratio'] = df['Myelocytes'] / df['Granulopoietic cells']
    df['Promyelocytes_to_Granulopoietic cells_Ratio'] = df['Promyelocytes'] / df['Granulopoietic cells']
    df['Myelocytes+Metamyelocytes_to_Granulopoietic cells_Ratio'] = (df[['Myelocytes', 'Metamyelocytes']].sum(axis=1)) / df['Granulopoietic cells']
  
    df['Metamyelocytes_to_Immature granulopoietic cells_Ratio'] = df['Metamyelocytes'] / df['Immature granulopoietic cells']
    df['Myelocytes_to_Immature granulopoietic cells_Ratio'] = df['Myelocytes'] / df['Immature granulopoietic cells']
    df['Promyelocytes_to_Immature granulopoietic cells_Ratio'] = df['Promyelocytes'] / df['Immature granulopoietic cells']

    if 'Eosinophils_Mature' in df.columns:
        df = df.drop(columns=["Eosinophils_Mature_Percentage", "Eosinophils_Mature"])
    
    return df


def create_features_for_tfr(df):
    
    df['Erythroid_cells'] = df[['Proerythroblasts', 'Erythroblasts']].sum(axis=1)
    df['Erythroid_cells_Proportion'] = 100 * df['Erythroid_cells'] / df['Living_Cells']

    df['Granulocytes'] = df[['Basophils', 'Eosinophils', 'Neutrophils', 'Metamyelocytes', 'Myelocytes', 'Promyelocytes']].sum(axis=1)
    df['Mature_Granulocytes'] = df[['Basophils', 'Eosinophils', 'Neutrophils']].sum(axis=1)
    df['Immature_Granulocytes'] = df[['Metamyelocytes', 'Myelocytes', 'Promyelocytes']].sum(axis=1)
    df['Granulocytes_Proportion'] = 100 * df['Granulocytes'] / df['Living_Cells']
    df['Mature_Granulocytes_Proportion'] = 100 * df['Mature_Granulocytes'] / df['Living_Cells']
    df['Immature_Granulocytes_Proportion'] = 100 * df['Immature_Granulocytes'] / df['Living_Cells']
    df['Neutrophils_Eosinophils_Proportion'] = 100 * df[['Eosinophils', 'Neutrophils']].sum(axis=1) / df['Living_Cells']

    df['Neutrophils_to_Basophils_Eosinophils_Ratio'] = df['Neutrophils'] / df[['Basophils', 'Eosinophils']].sum(axis=1)
    df['Mature_Granulocytes_to_All_Granulocytes_Ratio'] = df['Mature_Granulocytes'] / df['Granulocytes']
    df['Immature_Granulocytes_to_All_Granulocytes_Ratio'] = df['Immature_Granulocytes'] / df['Granulocytes']
    df['Mature_Granulocytes_to_Immature_Granulocytes_Ratio'] = df['Mature_Granulocytes'] / df['Immature_Granulocytes']

    df['Megakaryocytes_to_All_Granulocytes_Ratio'] = df['Megakaryocytes'] / df['Granulocytes']
    df['Erythroid_cells_to_All_Granulocytes_Ratio'] = df['Erythroid_cells'] / df['Granulocytes']
    df['Monocytes_to_All_Granulocytes_Ratio'] = df['Monocytes'] / df['Granulocytes']

    df['Neutrophils_to_Lymphocytes_Ratio'] = df['Neutrophils'] / df['Lymphocytes']
    df['Neutrophils_to_Lymphocytes_Ratio'] = df['Neutrophils_to_Lymphocytes_Ratio'].replace([np.inf, -np.inf], pd.NA)

    df['Mature_Granulocytes_to_Lymphocytes_Ratio'] = df['Mature_Granulocytes'] / df['Lymphocytes']
    df['Mature_Granulocytes_to_Lymphocytes_Ratio'] = df['Mature_Granulocytes_to_Lymphocytes_Ratio'].replace([np.inf, -np.inf], pd.NA)

    return df


def onehot_encode_with_nan(df, columns_to_encode, keep_original=False):
    # Make a copy of the dataframe to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # One-hot encode the specified columns
    df_encoded = pd.get_dummies(df_copy, columns=columns_to_encode, dummy_na=False)
    
    # Restore NaN values in the one-hot encoded columns based on the original DataFrame
    for column in columns_to_encode:
        na_mask = df[column].isna()
        encoded_columns = [col for col in df_encoded if col.startswith(column + '_')]
        df_encoded.loc[na_mask, encoded_columns] = pd.NA

        # Change dtype of one-hot encoded columns to bool
        for column in encoded_columns:
            df_encoded[column] = df_encoded[column].astype('boolean')  # Use 'boolean' dtype for nullable bool

    if keep_original:
        # Convert string values in the original columns to integer codes and overwrite the original columns
        for column in columns_to_encode:
            df_encoded[column] = df[column].astype('category').cat.codes
            df_encoded.loc[df[column].isna(), column] = pd.NA
    
    return df_encoded


def get_tfr_modelling_data(df_input, include_bcr_abl1_levels=False, onehot_encoded_version=False):

    df = df_input.copy()
    
    # Patient age (binary)
    df['Patient age at diagnosis > 50'] = df['Patient age at diagnosis > 50'].map({'yes': True, 'no': False}).astype('boolean')
    
    # spleen size (binary)
    df['Spleen size at diagnosis (max. distance from costal margin) > 0'] = df['Spleen size at diagnosis (max. distance from costal margin) > 0'].map({'yes': True, 'no': False}).astype('boolean')
        
    # transform binary string variables to binary integer variables
    df["Patient gender_male"] = df["Patient gender"].map({"male": True, "female": False}).astype('boolean')
    df = df.drop(columns="Patient gender")
    
    df['Relapse within 6 months'] = df['Relapse within 6 months'].map({"yes": True, "no": False}).astype('boolean')
    df['MR4.0 at 12 months'] = df['MR4.0 at 12 months'].map({"yes": True, "no": False}).astype('boolean')
    df['MMR at 12 months'] = df['MMR at 12 months'].map({"yes": True, "no": False}).astype('boolean')
    df['MMR at 6 months'] = df['MMR at 6 months'].map({"yes": True, "no": False}).astype('boolean')

    # EUTOS score
    df['EUTOS score-risk class_low'] = df['EUTOS score-risk class_low'].map({'yes': True, 'no': False}).astype('boolean')
    
    if not onehot_encoded_version:
        # Hasford, Sokal and ELTS score risk classes
        df = onehot_encode_with_nan(df, [
            "Patient age at diagnosis_categorized", 
            "Hasford score-risk class",
            "Sokal score-risk class", 
            "ELTS score-risk class"], True)
        
    # TKIs
    df[
            "First-line tyrosine kinase inhibitor is second-generation TKI"
        ] = df[
            "First-line tyrosine kinase inhibitor is second-generation TKI"
        ].map(
            {"yes": True, "no": False}
        ).astype('boolean')
    
    df[
            "Second-line tyrosine kinase inhibitor is second-generation TKI"
        ] = df[
            "Second-line tyrosine kinase inhibitor is second-generation TKI"
        ].map(
            {"yes": True, "no": False}
        ).astype('boolean')

    df[
            "Third-line tyrosine kinase inhibitor is second-generation TKI"
        ] = df[
            "Third-line tyrosine kinase inhibitor is second-generation TKI"
        ].map(
            {"yes": True, "no": False}
        ).astype('boolean')
    
    df[
            "Last tyrosine kinase inhibitor is second-generation TKI"
        ] = df[
            "Last tyrosine kinase inhibitor is second-generation TKI"
        ].map(
            {"yes": True, "no": False}
        ).astype('boolean')
    
    df[
            "Major tyrosine kinase inhibitor is second-generation TKI"
        ] = df["Major tyrosine kinase inhibitor is second-generation TKI"].map(
            {"yes": True, "no": False}
        ).astype('boolean')
        
    if onehot_encoded_version:
        # Patient age, Hasford, Sokal and ELTS score risk classes
        df = onehot_encode_with_nan(
            df, 
            [
                "Patient age at diagnosis_categorized",
                "Hasford score-risk class", 
                "Sokal score-risk class", 
                "ELTS score-risk class",
                "Last tyrosine kinase inhibitor generic name before discontinuation",
                "First-line tyrosine kinase inhibitor generic name before discontinuation",
                "Second-line tyrosine kinase inhibitor generic name before discontinuation",
                "Third-line tyrosine kinase inhibitor generic name before discontinuation",
                "Major tyrosine kinase inhibitor generic name before discontinuation",
            ], False)
    
    df["BCR-ABL <0.1% IS at 12 months"] = df[
        "BCR-ABL <0.1% IS at 12 months"
    ].map({"yes": True, "no": False}).astype('boolean')
    
    df["Relapse after first TKI discontinuation"] = df[
        "Relapse after first TKI discontinuation"
    ].map({"yes": True, "no": False}).astype(bool)
    
    df['Time to TKI discontinuation < 6 years'] = df['Time to TKI discontinuation < 6 years'].map({"yes": True, "no": False}).astype('boolean')

    # set Timepoint of possible relapse to be (current year (2023) - (year of diagnosis + Time to TKI discontinuation))*12 if
    # relapse has not occurred
    df[
        "Timepoint of possible relapse after first TKI discontinuation (months)"
    ] = df.apply(
        lambda row: (
            2024
            - (
                row["Year of diagnosis"]
                + row["Time to TKI discontinuation (months)"] / 12
            )
        )
        * 12
        - 2
        if pd.isna(
            row[
                "Timepoint of possible relapse after first TKI discontinuation (months)"
            ]
        )
        else row[
            "Timepoint of possible relapse after first TKI discontinuation (months)"
        ],
        axis=1,
    )

    # rows of these columns must be evaluated in case of string values (due to missing data in the clinical data table)
    columns_to_check_for_strings = [
        "Time to MMR (months)",
        "Time to MR4.0 (months)",
        "Time to MR4.5 (months)",
        "Duration of MMR before first TKI discontinuation (months)",
        "Duration of MR4.0 before first TKI discontinuation (months)",
        "Duration of MR4.5 before first TKI discontinuation (months)",
        "BCR-ABL1 transcript level at 3 months (%)",
        "BCR-ABL1 transcript level at 6 months (%)",
        "BCR-ABL1 transcript level at 12 months (%)",
    ]
    for column in columns_to_check_for_strings:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # create binary variables for achieving MMR, MR4.0 and MR
    df["MMR achieved"] = pd.notna(df["Time to MMR (months)"]).astype('boolean')
    df["MR4.0 achieved"] = pd.notna(df["Time to MR4.0 (months)"]).astype('boolean')
    df["MR4.5 achieved"] = pd.notna(df["Time to MR4.5 (months)"]).astype('boolean')

    # set Time to MR4.5 for patiens who have not achieved MR4.5
    df["Time to MR4.5 (months)"].fillna(
        df["Time to TKI discontinuation (months)"], inplace=True
    )
    df["Time to MR4.5 (months)"].fillna(0, inplace=True)

    # drop year of diagnosis
    df = df.drop(columns=["Year of diagnosis"])

    # Drop MR4.5 related columns for TFR prediction, because not all patients have achieved MR4.5-level
    df = df.drop(
        columns=[
            "Time to MR4.5 (months)",
            "Duration of MR4.5 before first TKI discontinuation (months)",
        ]
    )

    if "Hemavision ID" in df.columns.to_list():
        # drop Hemavision ID from the modelling data
        hemavision_ids = df["Hemavision ID"].to_numpy()
        df = df.drop(columns=["Hemavision ID"])
    else:
        hemavision_ids = df.index.to_numpy()

    return df, hemavision_ids
