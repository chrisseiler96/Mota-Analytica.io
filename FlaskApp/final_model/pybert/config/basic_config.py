#encoding:utf-8
from os import path
import multiprocessing
from pathlib import Path

#------------------------------------------------------------------------------------------------------------------
#----------------------------------------->>>>DATA PATHS>>>>-------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

BASE_DIR = Path('pybert')

configs = {

    'task':'multi label',
    'data':{
        'raw_data_path': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/dataset/raw/train.csv',  # 总的数据，一般是将train和test何在一起构建语料库
        'train_file_path': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/dataset/processed/train.tsv',
        'valid_file_path': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/dataset/processed/valid.tsv',
        'test_file_path': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/dataset/raw/test.csv'
    },
    'output':{
        'log_dir': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/output/log', # 模型运行日志
        'writer_dir': BASE_DIR / "/var/www/FlaskApp/FlaskApp/final_model/pybert/output/TSboard",# TSboard信息保存路径
        'figure_dir': BASE_DIR / "/var/www/FlaskApp/FlaskApp/final_model/pybert/output/figure", # 图形保存路径
        'checkpoint_dir': BASE_DIR / "/var/www/FlaskApp/FlaskApp/final_model/pybert/output/checkpoints",# 模型保存路径
        'cache_dir': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/model/',
        'result': BASE_DIR / "/var/www/FlaskApp/FlaskApp/final_model/pybert/output/result",
    },
    'pretrained':{
        "bert":{
            'vocab_path': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/model/pretrain/uncased_L-12_H-768_A-12/vocab.txt',
            'tf_checkpoint_path': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/model/pretrain/uncased_L-12_H-768_A-12/bert_model.ckpt',
            'bert_config_file': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/model/pretrain/uncased_L-12_H-768_A-12/bert_config.json',
            'pytorch_model_path': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/model/pretrain/pytorch_pretrain/pytorch_model.bin',
            'bert_model_dir': BASE_DIR / '/var/www/FlaskApp/FlaskApp/final_model/pybert/model/pretrain/pytorch_pretrain',
        },
        'embedding':{}
    },
    

#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------->>>>HYPERPARAMETERS>>>>---------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
  

    'train':{
        'valid_size': 0.2,
        'max_seq_len': 256,
        'do_lower_case':True,
        'batch_size': 8,#24,  # how many samples to process at once
        'epochs': 2,  # number of epochs to train
        'start_epoch': 1,
        'warmup_proportion': 0.1,# Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
        'gradient_accumulation_steps': 1,# Number of updates steps to accumulate before performing a backward/update pass.
        'learning_rate': 2e-5,
        'n_gpu': [0], # GPU个数,如果只写一个数字，则表示gpu标号从0开始，并且默认使用gpu:0作为controller,
                       # 如果以列表形式表示，即[1,3,5],则我们默认list[0]作为controller
        'num_workers': multiprocessing.cpu_count(), # 线程个数
        'weight_decay': 1e-5,
        'seed':42,
        'resume':False,
    },
    'predict':{
        'batch_size': 400
    },
    'callbacks':{
        'lr_patience': 5, # number of epochs with no improvement after which learning rate will be reduced.
        'mode': 'min',    # one of {min, max}
        'monitor': 'valid_loss',  # 计算指标
        'early_patience': 20,   # early_stopping
        'save_best_only': True, # 是否保存最好模型
        'save_checkpoint_freq': 10 # 保存模型频率，当save_best_only为False时候，指定才有作用
    },


#-------------------------------------------------------------------------------------------------------------------
#----------------------------------------->>>>MULTICLASS LABELS>>>>-------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
  
#Note: These must be the classes that are being predicted!
    'label2id' : { 
        "toxic": 0,
        "severe_toxicity": 1,
        "obscene": 2,
        "identity_attack": 3,
        "insult": 4,
        "threat": 5,
        "asian": 6,
        "atheist": 7,
        "bisexual": 8,
        "black": 9,
        "buddhist": 10,
        "christian": 11,
        "female": 12,
        "heterosexual": 13,
        "hindu": 14,
        "homosexual_gay_or_lesbian": 15,
        "intellectual_or_learning_disability": 16,
        "jewish": 17,
        "latino": 18,
        "male": 19,
        "muslim": 20,
        "other_disability": 21,
        "other_gender": 22,
        "other_race_or_ethnicity": 23,
        "other_religion": 24,
        "other_sexual_orientation": 25,
        "physical_disability": 26,
        "psychiatric_or_mental_illness": 27,
        "transgender": 28,
        "white": 29,
        "funny": 30,
        "wow": 31,
        "sad": 32,
        "likes": 33,
        "disagree":34,
        "sexual_explicit":35,
    },
    'model':{
        'arch':'bert'
    }
}