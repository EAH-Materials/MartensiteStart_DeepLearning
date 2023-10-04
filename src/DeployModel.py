import torch
from torch import nn, FloatTensor
from lightning import LightningModule

class DeployModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.define_model(config)
        self.eval()

        self.data_mean = {
            "C": 0.35294931950745295,
            "Mn": 0.7624457096565134,
            "Si": 0.332420194426442,
            "Cr": 1.1030602851587816,
            "Ni": 2.6435700129617627,
            "Mo": 0.29161202203499675,
            "V": 0.10516671419313027,
            "Co": 0.18455424497731693,
            "Al": 0.020678094620868438,
            "W": 0.37233430330524947,
            "Cu": 0.044665528191834086,
            "Nb": 0.00668602721970188,
            "Ti": 0.009008094620868438,
            "B": 9.397278029812057e-06,
            "N": 0.014538755670771227,
        }

        self.data_std = {
            "C": 0.2941571774906254,
            "Mn": 0.6501711392501798,
            "Si": 0.4029904606069103,
            "Cr": 2.211743845702123,
            "Ni": 6.342158791190849,
            "Mo": 0.6418696825408096,
            "V": 0.3884091859639032,
            "Co": 1.355922392821505,
            "Al": 0.14680902309859412,
            "W": 2.0992142449601774,
            "Cu": 0.17014241185441442,
            "Nb": 0.0931867931167249,
            "Ti": 0.11709327779176554,
            "B": 0.00015643096932446688,
            "N": 0.1416612051493458,
        }

        self.target_mean = 587.6420457809462
        self.target_std = 107.61116120718675
        self.order = ["C","Mn","Si","Cr","Ni","Mo","V","Co","Al","W","Cu","Nb","Ti","B","N"]

    # Building the model based on Checkpoint
    def define_model(self, config):
        last_out_size = config["layer1_output"]
        self.encoder = nn.Sequential(
            nn.Linear(config["input_shape"], last_out_size), 
            nn.LeakyReLU(),
        )

        dropout = config["layer1_dropout"]
        self.encoder.append(nn.Dropout(dropout))

        for layer_index in range(2, int(config["layer_count"])+1):
            next_out_size = config[f"layer{layer_index}_output"]
            self.encoder.append(nn.Linear(last_out_size, next_out_size))
            self.encoder.append(nn.LeakyReLU())
            dropout = config[f"layer{layer_index}_dropout"]
            self.encoder.append(nn.Dropout(dropout))
            last_out_size = next_out_size

        self.encoder.append(nn.Linear(last_out_size, 1))
    
    def forward(self, x):
        return self.encoder(x)
    
    # Inference methods
    def inference_pandas(self, df):
        self._scale_data_columns_pandas(df)
        with torch.no_grad():
            result = self(FloatTensor(df.values))
        result = self._rescale_target(result).detach().numpy().squeeze()
        return result
    
    def inference_vector(self, vec):
        self._scale_data_columns_vec(vec)
        with torch.no_grad():
            result = self(FloatTensor(vec))
        result = self._rescale_target(result).detach().numpy().squeeze()
        return result
    
    def inference_dict(self, dict):
        vec = self._compose_vector(dict)
        return self.inference_vector(vec)
    
    def _scale_data_columns_pandas(self, df):
        for column_name in df.columns:
            df[column_name] = self._scale(df[column_name], self.data_mean[column_name], self.data_std[column_name])
    
    def _scale_data_columns_vec(self, vec):
        for ix, val in enumerate(vec):
            vec[ix] = self._scale(val, self.get_nth_val(self.data_mean, ix), self.get_nth_val(self.data_std,ix))

    def _scale_target_columns(self, df):
        for column_name in df.columns:
            df[column_name] = self._scale_target_column(df, column_name)

    def _scale_target_column(self, df, column):
        return self._scale(
            df[column], self.target_mean[column], self.target_std[column]
        )
    
    def _rescale_target(self, value):
        return value*self.target_std + self.target_mean
        
    def _scale(self, value, mean, std):
        return (value - mean) / std
    
    def _compose_vector(self, composition_dict):
        vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for ix, key in enumerate(self.order):
            if key in composition_dict.keys():
                vector[ix] = composition_dict[key]

        return vector
    
    # Helper function to index into an ordered dict
    @staticmethod
    def get_nth_val(dictionary, n=0):
        """
        Get the nth value from an ordered dictionary.

        This static method allows you to retrieve the nth value from an ordered dictionary,
        optionally specifying the position using the 'n' parameter.

        Args:
            dictionary (OrderedDict or dict): An ordered dictionary or a regular dictionary.
                The order of retrieval is based on insertion order if using an OrderedDict.
            n (int, optional): The index of the value to retrieve. Default is 0, which returns the first value.

        Returns:
            object: The value at the specified index 'n' within the dictionary.

        Raises:
            IndexError: If the index 'n' is out of range or if the dictionary is empty.
        """
        if n < 0:
            n += len(dictionary)
        for i, val in enumerate(dictionary.values()):
            if i == n:
                return val
        raise IndexError("dictionary index out of range") 
