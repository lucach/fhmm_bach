## Bach HMM

#### Install
```bash
pip install -r requirements.txt
```
If you want to see and play music sheets generated from our scripts you have to install `musescore`
```bash
sudo apt install musescore
```
and uncomment following line in `music21_helpers.py` 
```python
music21.environment.set('musicxmlPath', '/usr/bin/musescore')
``` 

#### Usage
See the detailed help from argparse:
```bash
> python3 main.py --help
usage: main.py [-h] [--skip-hmmlearn] [--skip-fhmm] [--do-generation] [-K K]
               [-M M] [-max-iter MAX_ITER] [-training-size TRAINING_SIZE]

HMM / FHMM on Bach music.

optional arguments:
  -h, --help            show this help message and exit
  --skip-hmmlearn
  --skip-fhmm
  --do-generation
  -K K                  Size of hidden state alphabet.
  -M M                  Number of markov chains (for FHMM).
  -max-iter MAX_ITER    Maximum number of iterations during training.
  -training-size TRAINING_SIZE
                        Number of songs (absolute value) to use in the
                        training set, the remaining ones will be included in
                        the test set

```

There is also a demo you can run issuing `python3 demo.py`, after un-zipping
 `songs.zip`.