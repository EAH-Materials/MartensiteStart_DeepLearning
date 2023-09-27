import numpy as np
from sklearn.neighbors import LocalOutlierFactor

class LofMs():
    def __init__(self,df_data, space=[0,-1]):
        super().__init__()
        
        #df_ms = pd.read_csv(path)
        df_elements =  df_data[df_data.columns[space[0]:space[1]]].values

        Lof = LocalOutlierFactor(n_neighbors=16, 
                            algorithm='auto', 
                            leaf_size=30, 
                            metric='minkowski', 
                            p=2, 
                            metric_params=None, 
                            contamination='auto', 
                            novelty=True, 
                            n_jobs=None)


        self.lof = Lof.fit(df_elements)

    def dataset_lof(self):
        self.dataset_lof = -self.lof.negative_outlier_factor_
        return self.dataset_lof
    
    def score_samples(self,sample):
        if type(np.array([])) != type(sample):
            sample = np.array([sample])
        return -self.lof.score_samples(sample)



if __name__ == '__main__':
    print('start lof example')
    lof = LofMs(path='data/lof_space_ms.pkl')

    print(lof.dataset_lof())
    sam_scre = lof.score_samples(np.array([[1,0,0,0,5,1,0,0,0,0,1,0,0,0,0]]))
    print(sam_scre)