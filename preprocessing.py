import numpy as np
import pandas as pd

# Importing the dataset from Data folder and assigning it to a variable
neuron_df = pd.read_csv('Data/neurons.csv')
connection_df = pd.read_csv('Data/connections.csv')

non_conns = ['NO_CONS']
center_regions = ['NO', 'PB', 'EB', 'FB',  # Central Complex: Involved in multiple complex behaviors
                  'SAD', 'PRW',
                  # Periesophageal Neuropils Center: Sensory (including gustatory) processing, motor control, and potentially circadian rhythm.
                  'GNG',  # controlling the mouthparts, feeding behavior and taste info
                  'OCG',  # Ocelli: contribute to light detection and circadian rhythm
                  'UNASGD']  # Unassigned: no known function
left_right_regions = ['AME', 'LA', 'LO', 'LOP', 'ME',  # Optic lobe: Visual sensory region
                      'BU', 'LAL', 'GA',  # Lateral Complex: Integration of sensory information
                      'LH',  # Lateral Horn: Olfactory sensory region
                      'CAN', 'AMMC', 'FLA',  # Periesophageal Neuropils: Gustatory sensory region, motor control
                      'ICL', 'IB', 'ATL', 'CRE', 'SCL',  # Inferior Neuropils: Sensory integration, motor control
                      'VES', 'GOR', 'SPS', 'IPS', 'EPA',
                      # Ventromedial Neuropils: Sensory integration, learning and memory
                      'MB_PED', 'MB_VL', 'MB_ML', 'MB_CA',  # Mushroom Body: Olfactory learning and memory
                      'AL',  # Antennal Lobe: Olfactory sensory region
                      'SLP', 'SIP', 'SMP',  # Superior Neuropils: Integration of multi-modality sensory information
                      'AVLP', 'PVLP', 'WED', 'PLP',
                      'AOTU']  # Ventrolateral Neuropils: Sensory processing and cognitive functions

Exc_nt_type = ['ACH', 'DA', 'OCT', 'SER']
Inh_nt_type = ['GABA', 'GLU']

# Drop the columns that are not needed
neuron_df = neuron_df.drop(columns=["nt_type_score", "da_avg", "ser_avg", "gaba_avg", 'glut_avg', 'ach_avg', 'oct_avg'])

# Sort the data by the group
neuron_df = neuron_df.sort_values(by=['group'])

# Batch edit the value of group column based on regex: delete the string after the first dot
neuron_df['group'] = neuron_df['group'].str.replace(r'\..*', '')

# Rename the column name of connection_df pre_root_id to neuron_id
neuron_df = neuron_df.rename(columns={'root_id': 'neuron_id'})
connection_df = connection_df.rename(columns={'pre_root_id': 'neuron_id'})

neuron_conn_df = pd.merge(neuron_df, connection_df, on='neuron_id', how='right')
neuron_conn_df = neuron_conn_df.rename(columns={'group': 'pre_group'})
neuron_conn_df = neuron_conn_df.rename(columns={'nt_type_y': 'syn_type'})
neuron_conn_df = neuron_conn_df.rename(columns={'nt_type_x': 'nt_type'})

neuron_conn_indptr_df = pd.merge(neuron_df, connection_df, on='neuron_id', how='outer')
neuron_conn_indptr_df = neuron_conn_indptr_df.rename(columns={'group': 'pre_group'})
neuron_conn_indptr_df = neuron_conn_indptr_df.rename(columns={'nt_type_x': 'syn_type'})
neuron_conn_indptr_df = neuron_conn_indptr_df.rename(columns={'nt_type_y': 'nt_type'})

Neurons_dict = {}
Conn_dict = {}
for region in center_regions + left_right_regions:
  region_info = neuron_df.loc[neuron_df['group'] == region]
  region_info = region_info.sort_values(by=['neuron_id'])
  # delete all the duplicate neuron_id row
  region_neural_info = region_info.drop_duplicates(subset=['neuron_id'])
  # Neuron Information
  Neurons_dict[region] = {}
  Neurons_dict[region]['neuron_number'] = len(region_neural_info['neuron_id'])
  Neurons_dict[region]['neuron_ids'] = region_neural_info['neuron_id'].to_numpy()
  Neurons_dict[region]['nt_type'] = region_neural_info['nt_type'].to_numpy()

neuron_index = []
for i in center_regions + left_right_regions:
  neuron_index.append(Neurons_dict[i]['neuron_ids'])

neuron_index = np.concatenate(neuron_index).reshape(-1)


for region in center_regions + left_right_regions:
  # Connection Information
  region_conn_info = neuron_conn_df.loc[neuron_conn_df['pre_group'] == region]
  region_conn_info = region_conn_info.sort_values(by=['neuron_id'])
  Conn_dict[region] = {}
  Conn_dict[region]['conn_number'] = len(region_conn_info['neuron_id'])
  Conn_dict[region]['pre_ids'] = region_conn_info['neuron_id'].to_numpy()
  Conn_dict[region]['post_ids'] = region_conn_info['post_root_id'].to_numpy()
  Conn_dict[region]['syn_count'] = region_conn_info['syn_count'].to_numpy()
  region_conn_info['nt_type'] = region_conn_info['nt_type'].fillna(region_conn_info['syn_type'])
  Conn_dict[region]['syn_type'] = region_conn_info['nt_type'].to_numpy()

  # Indptr
  indptr_df = neuron_conn_indptr_df.loc[neuron_conn_indptr_df['pre_group'] == region]
  neuron_count = indptr_df['neuron_id'].value_counts()
  neuron_count = neuron_count.sort_index()
  neuron_count_array = neuron_count.to_numpy()
  for i in range(len(neuron_count)):
    if neuron_count_array[i] == 1 and \
        pd.isnull(indptr_df.loc[indptr_df['neuron_id'] == neuron_count.index[i]].syn_count).item():
      neuron_count_array[i] = 0
  Conn_dict[region]['indptr'] = neuron_count_array

  post_ids = region_conn_info['post_root_id'].to_numpy()
  post_index = np.asarray([np.argwhere(neuron_index == i)[0][0] for i in post_ids])
  Conn_dict[region]['post_index'] = post_index


import pickle

with open('Data/neuron.pkl', 'wb') as f:
  pickle.dump(Neurons_dict, f)

with open('Data/conn.pkl', 'wb') as f:
  pickle.dump(Conn_dict, f)
