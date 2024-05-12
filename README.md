<div align="center">
<img src=./image/logo.png width="250" />
</div>


<div align="center">
  <h2>Towards Cross-Table Masked Pretraining for Web Data Mining</h2>
</div>

<h3 align="center"> 
<a href="https://holyphd-my.sharepoint.com/:f:/g/personal/yc_holyphd_onmicrosoft_com/EvHXPwetLg5MgQX9tQLiFkcBxhPuOhGWx5VP6meMPq2FnA?e=3ci1QI"> 😀 Pretraining Model (CM2-v1) </a> |
<a href="https://holyphd-my.sharepoint.com/:f:/g/personal/yc_holyphd_onmicrosoft_com/EmlUjuSQhiFKo8PvvfrLSTQBZs2s_dTDFU47xb8CZpvvuQ?e=MsTDNL"> 📕 Pretraining Dataset (OpenTabs) </a> 
</h3>
</div>


## About our Work
Tabular data --- also known as structured data --- pervades the landscape of the World Wide Web, playing a foundational role in the digital architecture that underpins online information. Given the recent influence of large-scale pre-trained models like ChatGPT and SAM across various domains, exploring the application of pretraining techniques for mining tabular data on the web has emerged as a highly promising research direction. 
Indeed, there have been some recent works around this topic where most (if not all) of them are limited in the scope of a fixed-schema/single table. 
Due to the scale of the dataset and the parameter size of the prior models, we believe that we have not reached the ''BERT moment'' for the ubiquitous tabular data. The development on this line significantly lags behind the counterpart research domains such as natural language processing.
In this work, we first identify the crucial research challenges behind tabular data pretraining, particularly overcoming the cross-table hurdle. 
As a pioneering endeavor, this work mainly (i)-contributes a high-quality real-world tabular dataset, (ii)-proposes an innovative, generic, and efficient cross-table pretraining framework, dubbed as **CM2**, where the core to it comprises a semantic-aware tabular neural network that uniformly encodes heterogeneous tables without much restriction and (iii)-introduces a novel pretraining objective ---  Prompt Masked Table Modeling (pMTM) --- inspired from NLP but intricately tailored to scalable pretraining on tables. Our extensive experiments demonstrate CM2's state-of-the-art performance and validate that cross-table pretraining can enhance the performance of various downstream tasks.  

## How to Run

1. Install requirements.
```
conda create -n CM2 python=3.9
conda activate CM2
pip install -r requirements.txt
```

2. An example of fine-tuning using our pretrained model.
```
CUDA_VISIBLE_DEVICES=0 python -u run_finetune.py \
    --cpt ./CM2-v1 \
    --task_data ./example/cmc.csv,./example/car.csv \
```

3. An example of learning from scratch.
```
CUDA_VISIBLE_DEVICES=0 python -u run_scratch.py \
    --task_data ./example/cmc.csv,./example/car.csv \
```

4. An example of our prompt Masked Table Modeling (pMTM) pretraining.
```
deepspeed --master_port 29400 --num_gpus=4 run_mask_pretrain_ds.py \
    --deepspeed_config ds_config.json \
    --num_data 2500 \
    --num_epoch 2
```

## Datasets

### A New Cross-table Pretraining Dataset (OpenTabs)

You can download it from [here](https://holyphd-my.sharepoint.com/:f:/g/personal/yc_holyphd_onmicrosoft_com/EmlUjuSQhiFKo8PvvfrLSTQBZs2s_dTDFU47xb8CZpvvuQ?e=MsTDNL), and for more details about this dataset, you can find in the paper.  
🎉🎉🎉 **The works have used OpenTabs:**  
1. [Making Pre-trained Language Models Great on Tabular Prediction](https://openreview.net/pdf?id=anzIzGZuLi) (ICLR 2024)  
2. ...

### Downstream Task Datasets 
Breast https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original  
Cmc https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice   
Diabetes https://openml.org/d/37  
Vehicle https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes   
Satimage https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite  
Sick http://archive.ics.uci.edu/dataset/102/thyroid+disease  
Analcatdata https://pages.stern.nyu.edu/\~jsimonof/AnalCatData/Data/   
Pc1 https://openml.org/d/1068  
Adult https://archive.ics.uci.edu/dataset/2/adult   
PhishingWebsites https://archive.ics.uci.edu/dataset/327/phishing+websites   
Cylinder-bands https://archive.ics.uci.edu/dataset/32/cylinder+bands  
MiceProtein https://archive.ics.uci.edu/dataset/342/mice+protein+expression   
Car https://archive.ics.uci.edu/dataset/19/car+evaluation   
Segment http://archive.ics.uci.edu/dataset/50/image+segmentation   
Porto-seguro https://openml.org/d/44787  
Amazon https://openml.org/d/44712   
Elevators https://openml.org/d/216  
Yprop https://openml.org/d/416  
Topo https://openml.org/d/422   
SAT11 https://www.cs.ubc.ca/labs/algorithms/Projects/SATzilla/   
Diamonds https://openml.org/d/42225   
House_sales https://openml.org/d/42731   



## Reference Code
1. TransTab: https://github.com/RyanWangZf/transtab 