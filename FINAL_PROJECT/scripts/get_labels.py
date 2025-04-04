import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import wfdb
from tqdm import tqdm

DATA_DIR = '/Users/devleibowitz/Documents/TAU Courses/Deep Learning Raja/HW/dl_homeworks/FINAL_PROJECT/data'

# wide phenotype
def _phenotype_1(data_dir, record_id):
    try:
        record = wfdb.rdheader(os.path.join(data_dir, record_id))
        
        pH = apgar1 = apgar5 = nicu_days = gest_weeks = preeclampsia = None
        
        for comment in record.comments:
            if comment.startswith('pH'):
                pH = float(comment.split()[-1])
            elif comment.startswith('Apgar1'):
                apgar1 = int(comment.split()[-1])
            elif comment.startswith('Apgar5'):
                apgar5 = int(comment.split()[-1])
            elif comment.startswith('NICU days'):
                nicu_days = int(comment.split()[-1])
            elif comment.startswith('Gest. weeks'):
                gest_weeks = int(comment.split()[-1])
            elif comment.startswith('Preeclampsia'):
                preeclampsia = int(comment.split()[-1])
        
        is_compromised = (
            (pH is not None and pH < 7.2) or
            (apgar1 is not None and apgar1 < 7) or
            (apgar5 is not None and apgar5 < 7) or
            (nicu_days is not None and nicu_days > 0) or
            (gest_weeks is not None and gest_weeks < 37) or
            (preeclampsia is not None and preeclampsia == 1)
        )
        
        return 1 if is_compromised else 0
    
    except Exception as e:
        print(f"Error processing record {record_id}: {str(e)}")
        return None


# acidosis - only pH is considered
def _phenotype_2(data_dir, record_id):
    try:
        record = wfdb.rdheader(os.path.join(data_dir, record_id))
        
        pH = apgar1 = apgar5 = nicu_days = gest_weeks = preeclampsia = None
        
        for comment in record.comments:
            if comment.startswith('pH'):
                pH = float(comment.split()[-1])
        
        is_compromised = (
            (pH is not None and pH < 7.2)
        )
        
        return 1 if is_compromised else 0
    
    except Exception as e:
        print(f"Error processing record {record_id}: {str(e)}")
        return None


def label_all_samples(phenotype=1, output_file='labels.csv'):
    data_dir = os.path.join(DATA_DIR, "ctu_chb_data")
    hea_files = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.hea')]
    hea_files.sort()
    
    output_data = []
    
    # Process each .hea file with a progress bar
    for record_id in tqdm(hea_files, desc="Processing samples"):
        label = None
        if phenotype==1:
            label = _phenotype_1(data_dir, record_id)
        elif phenotype==2:
            label = _phenotype_2(data_dir, record_id)
        if label is not None:
            output_data.append([record_id, label])
    
    output_path = f'{DATA_DIR}/processed/phenotype_{phenotype}_{output_file}'
    # Write the output to a CSV file
    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['sample', 'label'])  # Write header
        csv_writer.writerows(output_data)
    
    print(f"Labeling complete. Output written to {output_path}")

    labels = pd.read_csv(output_path)
    print(labels['label'].value_counts())

label_all_samples(phenotype=2)