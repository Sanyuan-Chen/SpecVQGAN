
# mkdir /datablob/users/v-sanych/data/AudioSet/data/splited_19/
# split -dl 100000 /datablob/users/v-sanych/data/AudioSet/data/train.tsv /datablob/users/v-sanych/data/AudioSet/data/splited_19/train.tsv.
# for i in `seq -f "%02g" 01 18`; do sed -i '1i /datablob/users/v-chengw/data/AudioSet/data/' /datablob/users/v-sanych/data/AudioSet/data/splited_19/train.tsv.$i ; done
total=$1
idx=$2

#cp -r /modelblob/users/v-sanych/audioset_model/fairseq/finetune/esc50_fbank_mf_m16_02_mc16_02_audioset_fbank_patch_onl_hubert_pretrain_base_grep_dn_tin_mw1_uw0_mt28_lr5e4_fgm1_pm08_16_volp_iop16_p4/ /tmp/code/esc50_cp
#cd /tmp/code/esc50_cp/code
#sudo apt-get install build-essential -y
#/home/ubuntu/miniconda3/envs/specvqgan/bin/python -m pip list
#/home/ubuntu/miniconda3/envs/specvqgan/bin/python -m pip install --editable ./
#/home/ubuntu/miniconda3/envs/specvqgan/bin/python -m pip install sentencepiece h5py editdistance

cd /tmp/code/
echo "generating /datablob/users/v-sanych/data/AudioSet/data/splited_${total}/train.specvqgan_audioset_codebook_fbank_e1.${idx}"
/home/ubuntu/miniconda3/envs/specvqgan/bin/python extract_codes.py \
  2022-09-06T07-31-34_audioset_codebook_fbank \
  /modelblob/users/v-sanych/models/specvqgan_audioset_codebook_fbank_e1/ \
  /datablob/users/v-sanych/data/AudioSet/data/splited_${total}/train.tsv.${idx} \
  train.specvqgan_audioset_codebook_fbank_e1.${idx}

cp train.specvqgan_audioset_codebook_fbank_e1.${idx} /datablob/users/v-sanych/data/AudioSet/data/splited_${total}/train.specvqgan_audioset_codebook_fbank_e1.${idx}

