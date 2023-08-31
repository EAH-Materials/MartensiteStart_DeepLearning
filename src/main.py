import torch
from pandas import DataFrame
from DeployModel import DeployModel

if __name__ == "__main__":

    model = DeployModel.load_from_checkpoint("checkpoint/checkpoint", map_location=torch.device('cpu'))
    order = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N']
    values = [0, 0, 0, 0, 30.8, 0, 1.86, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Pandas Input
    test_pd = DataFrame(values, index=order).transpose()
    result = model.inference_pandas(test_pd)

    # Plain vector input
    result2 = model.inference_vector(values)

    print(result)
    print(result2)