import torch
import numpy as np

class MERT330M():
    def __init__(self, device):
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoModel

        self.sr = 24000
        self.device = device
        
        self.model = AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True)
        self.model.to(self.device)

    def get_embedding(self, audio, layer, aggregation):
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            hidden_states = out.hidden_states[layer]

            if aggregation == 'max':
                hidden_states = hidden_states.max(dim=1).values
                
            return hidden_states # [batch, 1024]