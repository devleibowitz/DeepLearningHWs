import os
import wfdb
import pandas as pd
import numpy
import json

DATA_DIR = "/Users/devleibowitz/Documents/TAU Courses/Deep Learning Raja/HW/dl_homeworks/FINAL_PROJECT/data"


def get_ehr(data_dir, label_file_path, output_path=os.path.join(DATA_DIR, "processed", "labeled_ehr.csv")):
    """Extracts EHR metrics from CTU-CHB dataset comments and returns a DataFrame."""

    records = sorted([f for f in os.listdir(data_dir) if f.endswith(".dat")], key=lambda x: int(x.split(".")[0]))
    
    ehr_data = []
    
    for record_file in records:
        record_id = record_file.split('.')[0]  # Extract numeric ID
        record_path = os.path.join(data_dir, record_id)
        
        try:
            record = wfdb.rdrecord(record_path)  # Read WFDB record
            comments = record.comments  # Extract comments
            
            ehr_entry = {"id": record_id}
            
            for comment in comments:
                parts = comment.split()
                if len(parts) == 2:  # Ensure it's a key-value pair
                    key, value = parts[0].lower(), parts[1]  # Normalize key names
                    try:
                        if value.strip().replace(".", "", 1).isdigit():  # Check if it's a valid number
                            value = float(value)  # Convert to float
                            if value.is_integer():  # Convert to int only if it's a whole number
                                value = int(value)
                        else:
                            value = 0  # Replace NaN with 0
                    except ValueError:
                        value = 0  # Handle non-numeric cases by setting to 0
                    ehr_entry[key] = int(value)  # Convert to integer
            
            ehr_data.append(ehr_entry)
        
        except Exception as e:
            print(f"Skipping {record_id} due to error: {e}")
    
    ehr_df = pd.DataFrame(ehr_data)

    # attach labels
    labels_df = pd.read_csv(label_file_path)

    # Ensure 'id' is of the same type in both dataframes
    labels_df['sample'] = labels_df['sample'].astype(str)  # Convert to string for consistency
    ehr_df['id'] = ehr_df['id'].astype(str)  

    # Merge the EHR dataframe with the labels
    df = ehr_df.merge(labels_df, left_on="id", right_on='sample', how="left")

    select_columns = [
        "id", "age", "gravidity", "parity", "diabetes", "hypertension",
        "preeclampsia", "pyrexia", "meconium", "presentation", "induced",
        "i.stage", "noprogress", "ck/kp", "ii.stage", "label"
            ]
    
    df = df[select_columns]
    df.to_csv(output_path)

    print(f"Labeled EHR table saved to: {output_path}.\n")
    return df

# Example usage:
data_dir = "/Users/devleibowitz/Documents/TAU Courses/Deep Learning Raja/HW/dl_homeworks/FINAL_PROJECT/data/ctu_chb_data"
label_file = '/Users/devleibowitz/Documents/TAU Courses/Deep Learning Raja/HW/dl_homeworks/FINAL_PROJECT/data/processed/phenotype_2_labels.csv'

ehr_df = get_ehr(data_dir, label_file)
print(ehr_df.head())
