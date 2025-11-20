from mlsolidpy.mlsolid import Mlsolid

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
