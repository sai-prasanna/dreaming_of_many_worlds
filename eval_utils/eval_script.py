import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='Load latents from dreamer and save them to a file')
    parser.add_argument('--latents_file', default='/work/dlclarge1/ramans-cmbrl/contextual_mbrl/logs/carl_classic_cartpole_single_0_enc_img_dec_img_pgm_ctx_normalized', type=str, help='Path to the latents file') #_pgm_ctx carl_classic_cartpole_single_0_enc_img_dec_img_normalized /work/dlclarge1/ramans-cmbrl/contextual_mbrl/logs/carl_classic_cartpole_double_box_enc_img_dec_img_pgm_ctx_adv_normalized
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    return parser.parse_args()


class LatentsDataset(Dataset):
    def __init__(self, latent_dir, state_keys = ['imagined',], target_key = ['context'], state_split ='all', transform=None):
        ''''
        data structured as follows
        .
        ├── seeds
        │   ├── ctx2latent.pkl

        in each ctx2latent.pkl file, there are list of 10 dicts with keys: 'context', 'episodes'
        each context is a dictionary with the following keys: 'gravity', 'length'
        each episodes is a list of 10 episodes, each episode is a dictionary with the following keys: ['image', 'imagined', 'obs', 'posterior']
        each of the keys in the episode dictionary is a numpy array of shape (T, d1, ..., dn) where T is the number of timesteps and d1, ..., dn are the dimensions of the data
        '''
        self.latent_dir = latent_dir
        if 'single' in self.latent_dir:
            match = re.search(r'single_(\d+)', self.latent_dir)
            print("match: ", match.group(1))
            self.target_keys = 'length' if int(match.group(1)) else 'gravity'
        else:
            self.target_keys = 'gravity'
        self.state_keys = state_keys
        self.state_split = state_split
        self.state_split_slice = slice(0, 512) if state_split == 'deter' else slice(513, 1536) if state_split == 'deter' else slice(0, 1536)
        #self.target_keys = target_keys
        self.transform = transform
        # load the data 
        for root, dirs, files in os.walk(latent_dir):
            for file in files:
                if file.endswith('ctx2latent_v1.pkl'):
                    #get seed number
                    seed = os.path.basename(root)
                    file_path = os.path.join(root, file)
                    data = load_file(file_path)
                    self.samples = []
                    self.targets = []
                    self.splits = []

                    for context_id, per_context_dict in enumerate(data):
                        #print("context_id: ", context_id, per_context_dict['context'])
                        context = per_context_dict['context'][self.target_keys]
                        for ep_id, episode in enumerate(per_context_dict['episodes']):
                            for key in self.state_keys:
                                samples_time = episode[key]
                                # input_data = samples_time.mean(axis=0)
                                # self.samples.append(input_data)
                                # self.targets.append(ep_id)
                                for timestep in range(len(samples_time)):
                                    input_data = samples_time[timestep][self.state_split_slice]
                                    self.samples.append(input_data)
                                    self.targets.append(context)
                                    self.splits.append(context_id % 3 > 1)
                                    
        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)
        self.splits = np.array(self.splits)

        #get unique targets and convert them to integers for classification instead of regression
        # self.unique_targets = np.unique(self.targets)
        # self.target_to_int = {target: i for i, target in enumerate(self.unique_targets)}
        # self.targets = np.array([self.target_to_int[target] for target in self.targets])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x = self.samples[idx]
        y = self.targets[idx]

        if self.transforms:
            x = self.transform(x)

        return x, y

def create_torch_dataset(latent_dir):
    latents_ds = LatentsDataset(latent_dir)
    return latents_ds


def visualize_tsne(latents_ds, save_dir=None):
    """
    Visualize the latents using t-SNE and color by the context
    latent_ds: LatentsDataset torch dataset
    """
    samples = latents_ds.samples
    targets = latents_ds.targets

    #print("samples shape: ", samples.shape)
    #print("targets shape: ", targets.shape)
    #perform t-SNE to visualize the data
    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(samples)
    print("kl-divergence: ", tsne.kl_divergence_)


    #plot the data using plotly
    fig = px.scatter_3d(x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2], color=targets)
    fig.update_layout(
    title="t-SNE visualization of latents, colour refers to context",
    xaxis_title="First t-SNE",
    yaxis_title="Second t-SNE",
    )
    fig.show()
    # save the plot in the same directory as the latents
    if save_dir:
        if not os.path.exists(os.path.join(save_dir, 'tsne_plots')):
            os.makedirs(os.path.join(save_dir, 'tsne_plots'))
        fig.write_html(os.path.join(save_dir, 'tsne_plots', os.path.basename(latents_ds.latent_dir) + '.html'))


def visualize_pca(latents_ds, save_dir=None):
    """
    Visualize the latents using PCA and color by the context
    latent_ds: LatentsDataset torch dataset
    """
    samples = latents_ds.samples
    targets = latents_ds.targets

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(samples)

    fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], color=targets)
    fig.update_layout(
    title="PCA visualization of latents, colour refers to context",
    xaxis_title="First PCA",
    yaxis_title="Second PCA",
    )
    fig.show()

    # save the plot in the save directory with the same name as the latents file 
    if save_dir:
        #make a directory for the pca plots if it does not exist
        if not os.path.exists(os.path.join(save_dir, 'pca_plots')):
            os.makedirs(os.path.join(save_dir, 'pca_plots'))
        fig.write_html(os.path.join(save_dir, 'pca_plots', os.path.basename(latents_ds.latent_dir) + '.html'))

def linear_probes(latents_ds, save_dir=None):
    """
    Train a linear probe on the latents and evaluate the performance
    latent_ds: LatentsDataset torch dataset
    """
    
    #create splits for train and test
    samples = latents_ds.samples
    targets = latents_ds.targets
    splits = latents_ds.splits

    X_train = samples[splits == 0]
    X_test = samples[splits == 1]
    y_train = targets[splits == 0]
    y_test = targets[splits == 1]
    #train a linear regression model using scikit-learn
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    #get the predictions
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)


    #create a csv file with the predictions and the true values
    df = pd.DataFrame({'true': y_test, 'predicted': y_pred})
    df['experiment'] = os.path.basename(latents_ds.latent_dir)
    #print("df: ", df.head(10))
    #evaluate the model
    
    #create a csv for the predictions and true values if save_dir is not None 
    if save_dir:
        path_preds = os.path.join(save_dir, 'linear_probe.csv')
        
        # if the file already exists, append to it
        if os.path.exists(path_preds):
            df.to_csv(path_preds, mode='a', header=False)
        else:
            df.to_csv(path_preds)

    #save score to a dataframe
    df = pd.DataFrame({'score': score, 'experiment': os.path.basename(latents_ds.latent_dir)}, index=[0])
    if save_dir:
        path_score = os.path.join(save_dir, 'linear_probe_scores.csv')
        if os.path.exists(path_score):
            df.to_csv(path_score, mode='a', header=False)
        else:
            df.to_csv(path_score)



def main():
    args = parse_args()
    print("Loading latents from file: ", args.latents_file)
    latents_ds = create_torch_dataset(args.latents_file)
    print("loaded latents from file: ", args.latents_file)
    save_dir = os.path.join(os.path.dirname(args.latents_file), 'latents_analysis')
    #make directory for analysis if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    

    print("Visualizing the latents using t-SNE and PCA ...")
    visualize_tsne(latents_ds, save_dir)
    visualize_pca(latents_ds, save_dir)
    print("Training a linear probe on the latents ...")
    linear_probes(latents_ds, save_dir)

    #visualize the data with TSN-E

    
if __name__ == "__main__":
    main()
    
