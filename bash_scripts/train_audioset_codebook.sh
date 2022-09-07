
#/home/ubuntu/miniconda3/envs/specvqgan/bin/python train.py --base configs/audioset_codebook_fbank.yaml \
#  -t True \
#  --gpus 0,1,2,3,4,5,6,7 \
#  --logdir /modelblob/users/v-sanych/models/specvqgan_audioset_codebook_fbank/ \
#  | tee -a /modelblob/users/v-sanych/models/specvqgan_audioset_codebook_fbank/log

cp -r /modelblob/users/v-sanych/audioset_model/fairseq/finetune/esc50_fbank_mf_m16_02_mc16_02_audioset_fbank_patch_onl_hubert_pretrain_base_grep_dn_tin_mw1_uw0_mt28_lr5e4_fgm1_pm08_16_volp_iop16_p4/ /tmp/code/esc50_cp
cd /tmp/code/esc50_cp/code
sudo apt-get install build-essential -y
/home/ubuntu/miniconda3/envs/specvqgan/bin/python -m pip list
/home/ubuntu/miniconda3/envs/specvqgan/bin/python -m pip install --editable ./
/home/ubuntu/miniconda3/envs/specvqgan/bin/python -m pip install sentencepiece h5py editdistance

cd /tmp/code/
model_path=/modelblob/users/v-sanych/models/specvqgan_audioset_codebook_fbank_fsp_esc50_patch_onl_hubert_p4
mkdir -p ${model_path}
/home/ubuntu/miniconda3/envs/specvqgan/bin/python train.py --base configs/audioset_codebook_fbank_fsp.yaml \
  -t True \
  --gpus 0,1,2,3,4,5,6,7 \
  --logdir ${model_path}/ \
  | tee -a ${model_path}/log

