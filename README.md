# Mlsolid-py
`mlsolid-py` is a python client for [mlsolid](https://github.com/zeddo123/mlsolid) MLOps platform.
This python SDK allows you to track experiments and save artifacts to the mlsolid server.
With Model Registries, you can easily version your machine learning models, and quickly access them in
your production code.

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

Model Registry example:
```Python
client = Mlsolid('localhost:5000')

created = client.create_model_registry('test_registry_1')

if not created:
    print('could not create model registry')
    exit()

run_id = None

with client.start_run('my_experiment') as run:
    run_id = run.run_id
    run.add_plaintext_artifact('./tests/data/plain_text_file.txt')
    run.add_model('./tests/data/mobile_sam.pt')

added = client.add_model('test_registry_1', run_id, 'mobile_sam.pt', ['latest'])
if not added:
    print('could not add model to registry')
    exit()

client.artifact(run_id, 'plain_text_file.txt')

client.tagged_model('test_registry_1', 'latest')

```
