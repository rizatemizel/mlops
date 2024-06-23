
import pickle
import pandas as pd
import sys
import os

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



year = int(sys.argv[1]) 
month =int(sys.argv[2]) 

source = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

df = read_data(source)


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


print("mean duration", y_pred.mean())



df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df['preds'] = y_pred

df_result = df[['ride_id','preds']]

output_directory = '/workspaces/mlops/04-deployment'
output_file = os.path.join(output_directory, f'df_resuts_{year:04d}-{month:02d}.pqt')

# Ensure the directory exists
os.makedirs(output_directory, exist_ok=True)

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)



