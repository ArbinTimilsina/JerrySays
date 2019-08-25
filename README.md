# Jerry Says

Try to predict how Jerry Seinfeld would finish a given incomplete sentence.

### Dataset for this project is obtained from https://www.kaggle.com/thec03u5/seinfeld-chronicles
and can be found in /seinfeld_scripts.

## Instructions

### Git clone the code and install it as a python library
```
git clone https://github.com/ArbinTimilsina/JerrySays.git
cd JerrySays
pip install -e .
```

### Download SpaCy english model
```
python -m spacy download en
```

### To train the model, do
```
train-jerry --help

# example
train-jerry -epoch 10 -batch_size 500
```

### To use the model, do
```
serve-jerry
```
and then curl the following (or such) in a new terminal (or paste into a web browser)
```
http://localhost:5050/autocomplete?seed=What+is+the+deal
```

You will get output similar to

<img src="plots/output.png" style="width: 500px;"/>

Replace the string after ```?seed=``` to change the seed and see suggested completions!


