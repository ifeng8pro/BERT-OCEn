# The Fine-Tuned BERT with One-Class Ensemble (BERRT-OCEn)


## Abstract
> > Code: This code implements a three-step pipeline: fine-tuning the BERT language model, performing one-class classification 
with an optimized CVDD algorithm, and integrating outputs through ensemble learning techniques, thereby effectively 
detecting fund misuse.
> 
 > > Dataset: This project provides a small `test` dataset at `data/corporra/test.xlsx`, for code execution testing and dataset format reference. 
It is also compatible with the public datasets  
>  > `Reuters-21578`(https://www.daviddlewis.com/resources/testcollections/reuters21578/),  
>  >  `20 Newsgroups`(http://qwone.com/~jason/20Newsgroups/),  
>  >  `IMDB Movie Reviews`(https://ai.stanford.edu/~amaas/data/sentiment/),  
>  > as well as the non-public datasets Dataset1 at  `data/corporra/yixiing.xlsx` and Dataset2 at `data/corporra/nanchang.xlsx`.They originate from a certain government agency and have undergone anonymization and partial processing.

## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`.


To run the code, we recommend setting up a virtual environment, e.g. using `conda`:

### `conda`
```
cd <path-to-BERT-OCEn-directory>
conda create --name myenv
source activate myenv
pip install -r requirements.txt
```



## Running experiments
You can run BERT-OCEn experiments using the `main.py` script.

Next, I will briefly explain how to run the `test` dataset.
You can run other datasets by replacing the dataset file read 
by `pandas` in `src/main.py` and `scr/datasets/***.py`.


### Dataset：[`test`]
Here is an example of running a dataset: `test` on a subset of data from dataset: `Dataset1`.using fine-tuned`bert-base-chinese` word embeddings for a BERT-OCEn model. 
```
cd <path-to-BERT-OCEn-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# change to source directory
cd src

# create folder for experimental output
mkdir ../log/test_yixing

# run experiment
 python main.py yixing cvdd_Net ../log/test_yixing ../data  --device cuda --seed 1 --clean_txt --embedding_size 768 --pretrained_model bert --ad_score context_dist_mean --n_attention_heads 4  --attention_size 300 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 --train_proportion 0.01 --finetune 1 --normal_class 100 --ensemble stacking;
```



Have a look into `main.py` for all the possible arguments and options.


