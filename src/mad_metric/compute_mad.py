import os
import gc

import librosa
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

import mauve

from .embedding_models import MERT330M

def featurize(input_dir, output_dir=None, batch_size=3, model_name='mert_330m', layer=24, aggregation='max'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'mert_330m':
        model = MERT330M(device)
    else:
        raise ValueError('Model name not supported')
    
    # Create output directory if it doesn't exist
    if output_dir != None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Embed
    batch_files = []
    batch_audios = []
    outs = []

    for idx, file in tqdm(enumerate(os.listdir(input_dir)), total=len(os.listdir(input_dir))):
        
        # Skip non-audio files
        if not file.endswith('.wav') and not file.endswith('.mp3'):
            continue

        # If the output file already exists, skip
        out_name = file.replace('.wav', '.pt').replace('.mp3', '.pt')
        if output_dir != None and os.path.exists(os.path.join(output_dir, out_name)):
            # Load the file and add it to the output
            emb = torch.load(os.path.join(output_dir, out_name))
            outs.append(emb)
            continue

        # Load audio
        file_path = os.path.join(input_dir, file)
        audio, _ = librosa.load(file_path, sr=model.sr)

        # Add to batch
        batch_files.append(file)
        batch_audios.append(audio)

        # Process batch
        if len(batch_audios) == batch_size or idx == len(os.listdir(input_dir)) - 1:
            batch_embs = model.get_embedding(batch_audios, layer, aggregation) # [batch, emb_dim]

            # Save the embeddings for each file
            if output_dir != None:
                for i, file in enumerate(batch_files):
                    out_name = file.replace('.wav', '.pt').replace('.mp3', '.pt')
                    torch.save(batch_embs[i, :,].unsqueeze(0), os.path.join(output_dir, out_name)) # [1, emb_dim]

            batch_files = []
            batch_audios = []

            outs.append(batch_embs.cpu())
    
    outs = torch.cat(outs, dim=0)

    # Free up memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return outs

def load_embs_from_dir(embs_dir):
    embs = []
    for file in os.listdir(embs_dir):
        emb = torch.load(os.path.join(embs_dir, file)) # [1, emb_dim]
        embs.append(emb)
    return torch.cat(embs, dim=0) # [num_files, emb_dim]

def compute_mad(
        eval_dir=None, 
        ref_dir=None, 
        eval_embs_dir=None, 
        ref_embs_dir=None, 
        log_csv=None,
        batch_size=3, 
        model_name='mert_330m', 
        layer=24, 
        aggregation='max'
        ):
    if eval_dir != None:
        eval_embs = featurize(eval_dir, eval_embs_dir, batch_size, model_name, layer, aggregation)
    elif eval_embs_dir != None:
        eval_embs = torch.load(eval_embs_dir)
    else:
        raise ValueError('Either eval_dir or eval_embs_dir must be provided')
    
    if ref_dir != None:
        ref_embs = featurize(ref_dir, ref_embs_dir, batch_size, model_name, layer, aggregation)
    elif ref_embs_dir != None:
        ref_embs = torch.load(ref_embs_dir)
    else:
        raise ValueError('Either ref_dir or ref_embs_dir must be provided')
    
    # Compute MAUVE
    mauve_out = mauve.compute_mauve(p_features=eval_embs, q_features=ref_embs)
    mauve_score = mauve_out.mauve
    # Take the negative log of the MAUVE score
    ln_mauve_score = - np.log(mauve_score)
    print(f'Input_dir: {eval_dir}, Ref_dir: {ref_dir}, Score: {ln_mauve_score}')

    # Log the results
    if log_csv != None:
        df = pd.DataFrame({
            'eval_dir': [eval_dir],
            'ref_dir': [ref_dir],
            'eval_embs_dir': [eval_embs_dir],
            'ref_embs_dir': [ref_embs_dir],
            'score': [ln_mauve_score]
        })

        if os.path.exists(log_csv):
            df.to_csv(log_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(log_csv, index=False)
    
    return ln_mauve_score