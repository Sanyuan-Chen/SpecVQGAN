'''
/home/ubuntu/miniconda3/envs/specvqgan/bin/python extract_codes.py 2022-09-06T07-31-34_audioset_codebook_fbank /modelblob/users/v-sanych/models/specvqgan_audioset_codebook_fbank/ train.tsv train.vqgan_fbank
'''
import os, sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from feature_extraction.demo_utils import load_model
from specvqgan.data.audioset import get_fbank

# model_name = '2022-09-06T06-57-48_audioset_codebook'
# log_dir = 'out_specvqgan/'
# split_path = 'train.tsv'
# save_path = 'train.vqgan_fbank'
model_name = sys.argv[1]
log_dir = sys.argv[2]
split_path = sys.argv[3]
save_path = sys.argv[4]
save_file = open(save_path, 'w')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# loading the models might take a few minutes
config, sampler, melgan, melception = load_model(model_name, log_dir, device)
sampler.eval()


class AudioSetFbank(torch.utils.data.Dataset):
    def __init__(self, config, split_path):
        super().__init__()
        self.sample_rate = config['data']['params']['sample_rate']
        self.fbank_bins = config['data']['params']['fbank_bins']
        self.fbank_mean = config['data']['params']['fbank_mean']
        self.fbank_std = config['data']['params']['fbank_std']

        self.dataset = open(split_path).read().splitlines()
        self.root = self.dataset[0]
        self.dataset = self.dataset[1:]

    def __getitem__(self, idx):
        length = int(self.dataset[idx].split()[1])
        if length < 400:
            return torch.zeros(1)
        vid = self.dataset[idx].split()[0]
        wav_path = os.path.join(self.root, vid)
        fbank = get_fbank(wav_path, output_sample_rate=self.sample_rate, n_bins=self.fbank_bins)
        mean, std = self.fbank_mean, self.fbank_std
        fbank = (fbank - mean) / std
        x = fbank.numpy().T
        x = torch.from_numpy(x)
        return x

    def __len__(self):
        return len(self.dataset)


dataset = AudioSetFbank(config, split_path)

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=6,
                        collate_fn=default_collate)

for x in tqdm(dataloader, total=len(dataset)):
    # vid = data.split()[0]
    # wav_path = os.path.join(root, vid)
    # fbank = get_fbank(wav_path, output_sample_rate=sample_rate, n_bins=fbank_bins)
    # mean, std = fbank_mean, fbank_std
    # fbank = (fbank - mean) / std
    # x = fbank.numpy().T
    # x = torch.from_numpy(x).unsqueeze(0).to(device)

    if x.numel() == 1 and x.item() == 0:
        save_file.write('\n')
        print(f"Detected too short audio")
        continue

    with torch.no_grad():
        x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float().to(device)
        quant_z, diff, info = sampler.encode(x)

        F, T = quant_z.shape[-2:]
        label = info[2].reshape(F, T).transpose(0, 1)
        label = label.reshape(-1).tolist()
        save_file.write(' '.join([str(l) for l in label])+'\n')

        # print(f"label: {label}")
        # xrec = sampler.decode(quant_z)
        # x = x[...,:xrec.shape[-1]]
        # # print(f'x: {x}; xrec: {xrec}; diff: {(xrec - x).abs()}')
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import Rectangle
        # def plot(matrix, file, labels=None):
        #     fig = plt.figure(figsize=(20, 5))
        #     heatmap = plt.pcolor(matrix.T)
        #     fig.colorbar(mappable=heatmap)
        #     plt.xlabel('Time(s)')
        #     ax = plt.gca()
        #     label_idx=0
        #     for x in range(0, matrix.shape[0], 16):
        #         for y in range(128-16, -16, -16):
        #             rect = Rectangle((x, y), 16, 16, linewidth=1, edgecolor='r', facecolor='none')
        #             ax.add_patch(rect)
        #             plt.text(x, y+8, str(labels[label_idx]), fontsize=5)
        #             label_idx += 1
        #     # plt.axis('square')
        #     # plt.ylabel(ylabel)
        #     plt.tight_layout()
        #     plt.savefig(file, dpi=300)
        # plot(x[0, 0].transpose(0, 1).cpu().detach().numpy(), f'/modelblob/users/v-sanych/audio_tmps/x.png', label)
        # plot(xrec[0, 0].transpose(0, 1).cpu().detach().numpy(), f'/modelblob/users/v-sanych/audio_tmps/xrec.png', label)
        # import pdb
        # pdb.set_trace()

