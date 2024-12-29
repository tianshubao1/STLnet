import numpy as np
import pandas as pd
from haversine import haversine, Unit
#from torch_geometric.utils import dense_to_sparse

#beijing_pm25 = torch.load('beijing_pm25.dat')
#beijing_pm25 = beijing_pm25.view(beijing_pm25.size(0), -1, 24)
#print(beijing_pm25.size())  #torch.Size([36, 365, 24])


beijing_data_train = np.load('beijing_data_new/train.npz')
beijing_data_train.files  #['x', 'y', 'x_offsets', 'y_offsets']
x_train = beijing_data_train['x'] #(2180, 24, 35, 6)
y_train = beijing_data_train['y'] #(2180, 24, 35, 6)

beijing_data_val = np.load('beijing_data_new/val.npz')
beijing_data_val.files  #['x', 'y', 'x_offsets', 'y_offsets']
x_val = beijing_data_val['x'] #(311, 24, 35, 6)
y_val = beijing_data_val['y'] #(311, 24, 35, 6)


beijing_data_test = np.load('beijing_data_new/test.npz')
beijing_data_test.files  #['x', 'y', 'x_offsets', 'y_offsets']
x_test = beijing_data_test['x'] #(623, 24, 35, 6)  seq_len:: 24
y_test = beijing_data_test['y'] #(623, 24, 35, 6)
#(3114, 24, 35, 6)
#print(x_test.shape)
#print(y_test.shape)

temp = x_train[100, :, :, :]  
#6: ['PM2.5', 'temperature', 'pressure', 'humidity', 'ws', 'wd']??
#6: ['PM2.5', 'temperature', 'humidity', 'pressure', 'ws', 'wd']

def get_adjacency_matrix(args):     #rewrite this?
    station_df = pd.read_csv(args)
    sensor_ids = station_df['station']
    num_sensors = len(station_df)

    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Calculate the Distance Matrix
    dist_mx = np.zeros((num_sensors, num_sensors))
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            coords1 = (station_df.loc[i, 'latitude'], station_df.loc[i, 'longitude'])
            coords2 = (station_df.loc[j, 'latitude'], station_df.loc[j, 'longitude'])
            distance = haversine(coords1, coords2, unit=Unit.KILOMETERS)
            dist_mx[i, j] = 1/distance
            dist_mx[j, i] = 1/distance

    # Apply threshold to adjacency matrix
    adj_mx = dist_mx.copy()
    print("Adjacency Matrix shape:", adj_mx.shape)

    # edge_index, dist = dense_to_sparse(torch.tensor(adj_mx))
    # edge_index, dist = edge_index.numpy(), dist.numpy()

    # def get_bearing(lat1, long1, lat2, long2):  #get angle 
    #     dLon = (long2 - long1)
    #     x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    #     y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1))* math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    #     brng = np.arctan2(x,y)
    #     brng = math.degrees(brng)
    #     brng = (brng + 360) % 360
    #     return brng

    # dist_arr = []
    # direc_arr = []



    return adj_mx


adj_mx = get_adjacency_matrix('beijing_data_new/station.csv')
np.save('adj_matrix_beiijng', adj_mx)
print(adj_mx.shape)



