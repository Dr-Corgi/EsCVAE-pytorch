# The EsCVAE model: A Pytorch implementation.
A Pytorch Implementation of EsCVAE: "[Learning to Converse Emotionally Like Humans: A Conditional Variational Approach](https://link.springer.com/content/pdf/10.1007%2F978-3-319-99495-6_9.pdf)" (Zhang R, Wang Z., CCF International Conference on Natural Language Processing and Chinese Computing 2018). A **Tensorflow** implementation can be found in [this repository](https://github.com/RuiZhang1993/EsCVAE-Tensorflow).

# Usage
Modify the *utils.py* to set the data\_path and run the *main.py* to run the model.  

More Details of the data format will be updated soon.

### Some Notes
1. Some details in this code is different from the paper. For example, we did not implement the beam search decoder here.
2. The hyper parameters are not fine-tuned in this implementation. Please modify the hyper parameters by yourself.

# Acknowledge
#### How do I cite EsCVAE?
```
@inproceedings{zhang2018learning,
    title={Learning to Converse Emotionally Like Humans: A Conditional Variational Approach},
    author={Zhang, Rui and Wang, Zhenyu},
    booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
    pages={98--109},
    year={2018},
    organization={Springer} 
}
```
