# Deep Learning Enabled Semantic Communication Systems

<center>Huiqiang Xie, Zhijin Qin, Geoffrey Ye Li, and Biing-Hwang Juang </center>

This is the implementation of  Deep learning enabled semantic communication systems.

## Requirements
```
tensorflow-gpu==2.1.0
bert4keras==0.4.2
```

## Preprocess

Preprocess the raw text to create the input sequences.

```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
python dataset/preprocess_text.py
```

## Train


The train option parameters are listed below

|    Parameters     |                           Help                           |
| :---------------: | :------------------------------------------------------: |
|    --train-snr    |                   Set the training SNR                   |
| --train-with-mine | Train with the MINE model to maximize mutual information |
|     --channel     |                 Set the training channel                 |
|       --bs        |                 The training batch size                  |
|       --lr        |              Set the training learning rate              |
| --checkpoint-path |                  The path to save model                  |

**Example:**

```shell
python main.py --bs=64 --train-snr=6 --channel=AWGN --train-with-mine --checkpoint-path=./checkpoint 
```

 

## Evaluation

The eval option parameters are listed below

|    Parameters     |              Help              |
| :---------------: | :----------------------------: |
|    --test-snr     |        Set the test SNR        |
|     --channel     |    Set the test channel    |
|       --bs        |    The training batch size     |
| --checkpoint-path |     The saved model     |

**Example:**

```shell
python evaluation.py --bs=256 --test-snr=6 --channel=AWGN --checkpoint-path=./checkpoint
```
### Notes
+ If you want to compute the sentence similarity, please download the BERT model.

## Bibtex
```bitex
@article{xie2021deep,
  author={H. {Xie} and Z. {Qin} and G. Y. {Li} and B. -H. {Juang}},
  journal={IEEE Transactions on Signal Processing}, 
  title={Deep Learning Enabled Semantic Communication Systems}, 
  year={Apr. 2021},
  volume={69},
  pages={2663-2675}}
```