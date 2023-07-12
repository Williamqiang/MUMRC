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

The dataset that we used in our experiments is as follows:



+ MRE
    
    The MRE dataset comes from [MEGA](https://github.com/thecharm/Mega), many thanks.

    You can download the **MRE dataset with detected visual objects** from [Google Drive](https://drive.google.com/file/d/1q5_5vnHJ8Hik1iLA9f5-6nstcvvntLrS/view?usp=sharing) or using following command:
    
    ```bash
    cd MRE
    wget 120.27.214.45/Data/re/multimodal/data.tar.gz
    tar -xzvf data.tar.gz
    ```


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

    ```

+ ## MRE Task

    To run mre task, run this script.

    ```shell
    cd MRE
    bash run_mre.sh
    ```

# Acknowledgement

The acquisition of image data for the multimodal link prediction task refer to the code from [https://github.com/wangmengsd/RSME](https://github.com/wangmengsd/RSME), many thanks.

# Papers for the Project & How to Cite
If you use or extend our work, please cite the paper as follows:

```bibtex
@inproceedings{DBLP:conf/sigir/ChenZLDTXHSC22,
  author    = {Xiang Chen and
               Ningyu Zhang and
               Lei Li and
               Shumin Deng and
               Chuanqi Tan and
               Changliang Xu and
               Fei Huang and
               Luo Si and
               Huajun Chen},
  editor    = {Enrique Amig{\'{o}} and
               Pablo Castells and
               Julio Gonzalo and
               Ben Carterette and
               J. Shane Culpepper and
               Gabriella Kazai},
  title     = {Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge
               Graph Completion},
  booktitle = {{SIGIR} '22: The 45th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval, Madrid, Spain, July 11 -
               15, 2022},
  pages     = {904--915},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3477495.3531992},
  doi       = {10.1145/3477495.3531992},
  timestamp = {Mon, 11 Jul 2022 12:19:20 +0200},
  biburl    = {https://dblp.org/rec/conf/sigir/ChenZLDTXHSC22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
