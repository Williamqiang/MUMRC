# MUMRC

Code for the ICME 2023 paper "[A Unified MRC Framework with Multi-Query for Multi-modal Relation Triplets Extraction]"

# Model Architecture

<div align=center>
<img src="model.png" width="75%" height="75%" />
</div>
 
 
 Illustration of MUMRC.


# Requirements


```
conda create -n mumrc python==3.7
conda activate mumrc
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install transformers==4.18.0
```


# Data Download


    The  dataset comes from [MEGA](https://github.com/thecharm/Mega) and [MKGformer](https://github.com/zjunlp/MKGformer), many thanks.


The expected structure of files is:


```
MKGFormer
 |-- MKG	# Multimodal Knowledge Graph
 |    |-- dataset       # task data
 |    |-- data          # data process file
 |    |-- lit_models    # lightning model
 |    |-- models        # mkg model
 |    |-- scripts       # running script
 |    |-- main.py   
 |-- MNER	# Multimodal Named Entity Recognition
 |    |-- data          # task data
 |    |    |-- twitter2017
 |    |    |    |-- twitter17_detect            # rcnn detected objects
 |    |    |    |-- twitter2017_aux_images      # visual grounding objects
 |    |    |    |-- twitter2017_images          # raw images
 |    |    |    |-- train.txt                   # text data
 |    |    |    |-- ...
 |    |    |    |-- twitter2017_train_dict.pth  # {imgname: [object-image]}
 |    |    |    |-- ...
 |    |-- models        # mner model
 |    |-- modules       # running script
 |    |-- processor     # data process file
 |    |-- utils
 |    |-- run_mner.sh
 |    |-- run.py
 |-- MRE    # Multimodal Relation Extraction
 |    |-- data          # task data
 |    |    |-- img_detect   # rcnn detected objects
 |    |    |-- img_org      # raw images
 |    |    |-- img_vg       # visual grounding objects
 |    |    |-- txt          # text data
 |    |    |    |-- ours_train.txt
 |    |    |    |-- ours_val.txt
 |    |    |    |-- ours_test.txt
 |    |    |    |-- mre_train_dict.pth  # {imgid: [object-image]}
 |    |    |    |-- ...
 |    |    |-- vg_data      # [(id, imgname, noun_phrase)], not useful
 |    |    |-- ours_rel2id.json         # relation data
 |    |-- models        # mre model
 |    |-- modules       # running script
 |    |-- processor     # data process file
 |    |-- run_mre.sh
 |    |-- run.py
```

# How to run
  ```shell
  cd MUMRC_BERT
  bash train.sh
  ```

# Acknowledgement

The code is based on [PURE](https://github.com/princeton-nlp/PURE), many thanks.

# Papers for the Project & How to Cite
If you use or extend our work, please cite the paper as follows:

```bibtex
dwadwa dadwd
```
