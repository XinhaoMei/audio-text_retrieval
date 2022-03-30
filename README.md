# On Metric Learning for Audio-Text Cross-Modal Retrieval

## Set up environment

* Clone the repository: `git clone https://github.com/XinhaoMei/audio-text_retrieval.git`
* Create conda environment with dependencies: `conda env create -f environment.yaml -n name`
* All of our experiments are running on RTX 3090 with CUDA11. This environment just works for RTX 30x GPUs.

## Set up dataset 

* AudioCaps can be downloaded at https://github.com/XinhaoMei/ACT,
* Clotho can be downloaded at https://zenodo.org/record/4783391#.YkRHxTx5_kk.
* Unzip downloaded files, and put the wav files under `data/AudioCaps/waveforms` or `data/Clotho/waveforms`, the folder structured like
```
  data
  ├── AudioCaps
  │   ├── csv_files  
  │   ├── waveforms
  │      ├── train
  │      ├── val
  │      ├── test
  ├── Clotho
  │   ├── csv_files  
  │   ├── waveforms
  │      ├── train
  │      ├── val
  │      ├── test
  
  ```

## Pre-trained encoders
Pre-trained audio encoders CNN14 and ResNet38 can be downloaded at: https://github.com/qiuqiangkong/audioset_tagging_cnn

### Run experiments
* Set the parameters you want in `settings/settings.yaml` 
* Run experiments: `python train.py -n exp_name`

## Cite

If you use our code, please kindly cite following:
```
@article{Mei2022metric,
  title = {On Metric Learning for Audio-Text Cross-Modal Retrieval},
  author = {Mei, Xinhao and Liu, Xubo and Sun, Jianyuan and Plumbley, Mark D. and Wang, Wenwu},
  journal={arXiv preprint arXiv:2203.15537},
  year={2022}
}

```
and
```
@inproceedings{Mei2021ACT,
    author = "Mei, Xinhao and Liu, Xubo and Huang, Qiushi and Plumbley, Mark D. and Wang, Wenwu",
    title = "Audio Captioning Transformer",
    booktitle = "Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)",
    address = "Barcelona, Spain",
    month = "November",
    year = "2021",
    pages = "211--215",
    isbn = "978-84-09-36072-7",
    doi. = "10.5281/zenodo.5770113"
}
```