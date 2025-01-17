import pickle

data_path = "data/WaterRamps/test/0.pkl"

with open(data_path, 'rb') as f:
    data = pickle.load(f)

for key, value in data.items():
    print(key, value.shape)

print(data["particle_type"])

for i in range(10):
    data_path = f"data/FEM/train/{i}.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # find the max and min of pos in x, y, z coordinates
    print(f"Max pos in x: {data['position'][:, :, 0].max()}, min pos in x: {data['position'][:, :, 0].min()}")
    print(f"Max pos in y: {data['position'][:, :, 1].max()}, min pos in y: {data['position'][:, :, 1].min()}")
    
    count5 = 0
    count3 = 0
    print(f"Check {data_path} now...")
    for i in range(data["particle_type"].shape[0]):
        if data["particle_type"][i] == 5:
            count5 += 1
        elif data["particle_type"][i] == 3:
            count3 += 1
        else:
            print(f"Found particle type {data['particle_type'][i]}")
    print(f"There are {count5} particles of type 5 and {count3} particles for type 3 for {data_path}")
    print(f"There are total {count5 + count3} particles for {data_path}")
    print(f"The data is of shape {data['position'].shape}\n")