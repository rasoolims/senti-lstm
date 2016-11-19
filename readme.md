# Cross-lingual Sentiment Analysis with Deep Learning

## Requirement
You need to install [dynet](https://github.com/clab/dynet) library and its python wrapper. Look at its [installation guidelines](http://dynet.readthedocs.io/en/latest/python.html). 

__Note:__ it is usually recommended to use [virtualenv](https://virtualenv.pypa.io/en/stable/) to install python-based libraries.


## Current functionalities

### Train
Train data should contain one sentence at each line with labels separated by __tab__. Each word can have translation separated by __|||__ (translation is optional). Labels can be anything. The system saves the best performing model based on the development data.

```
python -u src/sentilstm.py --train [train-file] --dev [development-file] --outdir [output-directory]--embed [embedding-file(word2vec format))] 
```

To see all options type:
```
python -u src/sentilstm.py --help
```

The best model is saved at [output-directory]/model.model and its parameter file as [output-directory]/params.pickle. The development data is also optional and the model file for each epoch is saved at [output-directory]/model.model_iter_3,[output-directory]/model.model_iter_1, [output-directory]/model.model_iter_2; etc.

### Run Blind Data

Train data should contain one sentence at each line (they can have labels as well but the code does not look at them).

```
python -u src/sentilstm.py  --input [input_file] --output [output_file] --model [model_file] --params [params_file] 
```