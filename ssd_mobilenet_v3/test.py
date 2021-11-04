import numpy as np
import pandas as pd
from torchreid.utils.feature_extractor import FeatureExtractor

feature_model = 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
query_data = 'data\query_data.csv'

extractor = FeatureExtractor(
    model_name='osnet_x1_0', model_path=feature_model, device='cpu'
)

img = [
    'data\query\query0.jpg', 'data\query\query1.jpg', 'data\query\query2.jpg',
    'data\query\query3.jpg', 'data\query\query4.jpg', 'data\query\query5.jpg'
]

# g_data = pd.read_csv('data\query_data.csv')
# g_data.insert(0, 'id', 5)
# print(g_data)
# a = g_data.replace({'id': 5}, 2)
# print(a)
# features = pd.DataFrame(extractor(img).numpy())
# features.to_csv(path_or_buf=query_data, index=False)
# print(features.iloc[0])


a = [(0, 11), (1, 40), (2, 6)]
print(max(a,key=lambda i:i[1])[0])