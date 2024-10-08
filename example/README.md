### Overview

This is a minimal example showcasing a config and a dataset. Things to note:
- config file accepts CometML credentials for tracking the training run (optional);
- config files accepts multiple paths for each of the datasets. Meaning you can generate multiple batches of your data stored in different folders. Each folder must contain *.jsonl files with a similar structure to the example files.

### Running

Take the script from the `script` folder and run:

- `bash run.sh config.yaml`

The bash script will create a `logs` folder with a `config.yaml.log` file containing all outputs during the training process. The library itself will create an `output` folder (as specified in the config) and save a copy of the config there (without CometML credentials). Over the training run the best checkpoint based off validation metrics will be saved there.