import argparse
import torch
from B.train import train
from B.test import test
from B.model import ABSABert
from B.data import data_preprocessing
from B.loss import ACD_loss, ACSA_loss

def run(args):
    

    # ======================================================================================================================
    # Data preprocessing
    data_train, data_val, data_test = data_preprocessing(args, 'b')
    # ======================================================================================================================
    # Task B
    model_B = ABSABert(args)
    acc_B_train = train(model_B, data_train, data_val, ACSA_loss, args)
    acc_B_test = test(model_B, data_test, args)
    # Clean up memory/GPU etc...
    torch.cuda.empty_cache()


    # ======================================================================================================================
    ## Print out your results with following format:
    print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
                                                            acc_B_train, acc_B_test))

    # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
    # acc_A_train = 'TBD'
    # acc_B_test = 'TBD'

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='Datasets/Restaurants_Train.xml')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=5)

    args = parser.parse_args()

    run(args)