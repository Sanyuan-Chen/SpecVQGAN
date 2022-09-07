'''
cp -r /modelblob/users/v-sanych/audioset_model/fairseq/finetune/esc50_fbank_mf_m16_02_mc16_02_audioset_fbank_patch_onl_hubert_pretrain_base_grep_dn_tin_mw1_uw0_mt28_lr5e4_fgm1_pm08_16_volp_iop16_p4/ /tmp/code/esc50_cp
cd  /tmp/code/esc50_cp/code
sudo apt-get install build-essential -y
/home/ubuntu/miniconda3/envs/specvqgan/bin/python -m pip list
/home/ubuntu/miniconda3/envs/specvqgan/bin/python -m pip install --editable ./
/home/ubuntu/miniconda3/envs/specvqgan/bin/python -m pip install sentencepiece h5py editdistance

# /home/ubuntu/miniconda3/envs/specvqgan/bin/python setup.py build_ext --inplace
# /home/ubuntu/miniconda3/envs/specvqgan/bin/python setup.py install --user
'''

import torch
import fairseq
import torch.nn as nn

import sys
sys.path.insert(0, '.')  # nopep8


class LPAPS_fairseq(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        pretrained_ckpt = "/tmp/code/esc50_cp/checkpoint_best.pt"
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([pretrained_ckpt])
        print("loaded pretrained esc50 model from {}".format(pretrained_ckpt))
        self.model = model[0]
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # print(f"input: {input.shape}, target: {target.shape}")  # B, 1, F, T
        # self.model.w2v_encoder.w2v_model.encoder.layers[-1].fc2.weight
        B = input.shape[0]
        combine = torch.cat((input, target), dim=0).squeeze(1).transpose(1, 2)
        combine_out = self.model(source=combine, padding_mask=torch.zeros_like(combine), fbank=combine, tbc=False)['encoder_out'].unsqueeze(1).transpose(2, 3)
        diff = (combine_out[:B] - combine_out[B:]) ** 2
        diff = spatial_average(diff, keepdim=True)
        return diff


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


if __name__ == '__main__':
    inputs = torch.rand((16, 1, 128, 992))
    reconstructions = torch.rand((16, 1, 128, 992))
    lpips = LPAPS_fairseq().eval()
    loss_p = lpips(inputs.contiguous(), reconstructions.contiguous())
    # (16, 1, 1, 1)
    print(loss_p.shape)
