# Mlsolid-py
`mlsolid-py` is a python client for [mlsolid](https://github.com/zeddo123/mlsolid) MLOps platform.
This python SDK allows you to track experiments and save artifacts to the mlsolid server.

## Usage
`mlsolid` provides an easy way to track your experiment runs:
```Python
from mlsolidpy.mlsolid import Mlsolid

client = Mlsolid('localhost:5000')

print('Experiments', client.experiments)

print('Run ', client.run("urbane-wagon"))

with client.start_run('my_experiment') as run:
    run.log({'checkpoint': "path/to/checkpoint"})
    run.log({'batch-size': 23})

    run.log({'mae': 0.2333, 'loss': 100.0})
    run.log({'mae': 0.2000, 'loss': 90})
    run.log({'mae': 0.1134, 'loss': 10})
    run.log({'metrics': [0.2000, 0.333, 0.2223]})
```
