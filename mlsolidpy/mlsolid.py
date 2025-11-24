from collections import defaultdict
import pathlib
from typing import Any, Iterator, Dict
from contextlib import contextmanager
from pathlib import Path

import grpc
import randomname
from rich.console import Console
from rich.progress import track
from mlsolid.v1 import mlsolid_pb2_grpc
from mlsolid.v1.mlsolid_pb2 import AddArtifactRequest, AddMetricsRequest, AddModelEntryRequest, AddModelEntryResponse, ArtifactRequest, ArtifactResponse, Content, CreateModelRegistryRequest, CreateModelRegistryResponse, CreateRunRequest, ExperimentRequest, ExperimentResponse, ExperimentsRequest, MetaData, ModelEntry, ModelRegistryRequest, ModelRegistryResponse, RunRequest, RunResponse, StreamTaggedModelRequest, StreamTaggedModelResponse

from mlsolidpy.exceptions import BadRequest, NotFound, InternalError
from mlsolidpy.types import Artifact, ArtifactType, FloatMetric, Metric, ModelRegistry, Run, StrMetric, chunk_bytes, from_protobuf_metrics, to_protobuf_metrics


console = Console()

class RunManager:
    def __init__(self, run_id: str, exp_id: str) -> None:
        self.exp_id = exp_id
        self.run_id = run_id
        self._metrics = defaultdict(list)
        self._artifacts = []

    @property
    def metrics(self) -> list[Metric]:
        return [self._parse_metric(name, vals) for name, vals in self._metrics.items()]

    @property
    def artifacts(self) -> list[Artifact]:
        return self._artifacts

    def log(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            self._metrics[k].append(v)

    def add_model(self, path: str) -> None:
        name, content = self._read_file(path)

        self._add_artifact(Artifact(name=name,
                                   artifact_type=ArtifactType.ModelArtifact.value,
                                   run_id=self.run_id,
                                   content=content))

    def add_plaintext_artifact(self, path: str) -> None:
        name, content = self._read_file(path)

        self._add_artifact(Artifact(name=name,
                                   artifact_type=ArtifactType.PlainTextArtifact.value,
                                   run_id=self.run_id,
                                   content=content))

    def _add_artifact(self, artifact: Artifact):
        self._artifacts.append(artifact)

    def _read_file(self, path: str):
        path_obj = Path(path)


        if not path_obj.exists() or not path_obj.is_file():
            raise Exception(f'`{path}` does not exist')

        with open(path_obj, 'rb') as f:
            b = f.read()

            return path_obj.name, b

    def _parse_metric(self, name: str, vals: list[Any]) -> Metric:
        flat = []
        for val in vals:
            if type(val) in (list, tuple):
                flat.extend(val)
            else:
                flat.append(val)

        types = set((float if type(val) is int else type(val) for val in flat))

        if len(types) > 1:
            return StrMetric(name=name, vals=[str(val) for val in vals])
        else:
            t = types.pop()
            if t is float:
                return FloatMetric(name=name, vals=flat)
            else:
                return StrMetric(name=name, vals=flat)

class Mlsolid:
    def __init__(self, url: str, insecure: bool = False, ca_cert_path: None | str = None) -> None:
        self.url = url

        if insecure:
            self.channel = grpc.insecure_channel(self.url)
        elif ca_cert_path is None:
            self.channel = grpc.secure_channel(self.url, grpc.ssl_channel_credentials())
        elif ca_cert_path:
            with open(ca_cert_path) as f:
                root_certificates = f.read()
                self.channel = grpc.secure_channel(self.url, grpc.ssl_channel_credentials(
                    root_certificates=root_certificates
                    ))

        self.stub = mlsolid_pb2_grpc.MlsolidServiceStub(self.channel)

    @property
    def experiments(self) -> list[str]:
        res = self.stub.Experiments(ExperimentsRequest())

        return [id for id in res.exp_ids]

    def experiment(self, exp_id: str) -> list[str]:
        try:
            res : ExperimentResponse = self.stub.Experiment(ExperimentRequest(exp_id=exp_id))

            return [id for id in res.run_ids]
        except grpc.RpcError as e:
            raise self._handle_grpc_error(e)

    def run(self, id: str) -> Run | None:
        try:
            resp : RunResponse = self.stub.Run(RunRequest(run_id=id))

            return Run(
                    id=resp.run_id,
                    timestamp=resp.timestamp.ToDatetime(),
                    exp_id=resp.experiment_id,
                    metrics=from_protobuf_metrics(resp.metrics.values()) #type: ignore
                    )
        except grpc.RpcError as e:
            raise self._handle_grpc_error(e)
        except Exception as e:
            raise e

    def create_run(self, run_id: str, exp_id: str):
        """
        create_run creates a run with a fixed run_id.
        If the run_id exists, a BadRequest is raised.
        """
        try:
            self.stub.CreateRun(CreateRunRequest(run_id=run_id, experiment_id=exp_id))
        except grpc.RpcError as e:
            raise self._handle_grpc_error(e)

    def new_run(self, exp_id: str) -> str:
        """new_run creates a new run and returns its id"""
        exception = Exception()

        for _ in range(10):
            try:
                id = randomname.get_name()
                self.create_run(run_id=id, exp_id=exp_id)
                return id
            except Exception as e:
                exception = e

        raise exception

    @contextmanager
    def start_run(self, exp_id: str, dry : bool = False) -> Iterator[RunManager]:
        if not dry:
            run_id = self.new_run(exp_id)
        else:
            run_id = randomname.get_name()
        
        console.print(f':test_tube: Started run {run_id} :: {exp_id}', style='bold green')

        run = RunManager(run_id, exp_id)

        yield run

        if not dry:
            self._commit_run(run)

    def add_metrics(self, run_id: str, metrics: list[Metric]):
        try:
            self.stub.AddMetrics(AddMetricsRequest(run_id=run_id, metrics=to_protobuf_metrics(metrics)))

            console.print('📥 Metrics uploaded to mlsolid', metrics, style='bold')
        except grpc.RpcError as e:
            raise self._handle_grpc_error(e)

    def artifact(self, run_id: str, artifact_name: str, path='./.mlsolid/artifacts'):
        try:
            resp: ArtifactResponse = self.stub.Artifact(ArtifactRequest(run_id=run_id, artifact_name=artifact_name))

            metadata: MetaData = MetaData()
            content = bytearray()

            for response in track(resp, description='Downloading'):
                one = response.WhichOneof('request')
                if one == 'metadata':
                    metadata = response.metadata
                elif one == 'content':
                    content.extend(response.content.content)

            self._save_artifact(Artifact(name=metadata.name,
                                         artifact_type=ArtifactType.ModelArtifact.value,
                                         run_id=metadata.run_id,
                                         content=bytes(content)), path)
        except grpc.RpcError as e:
            raise self._handle_grpc_error(e)

    def add_artifacts(self, run_id: str, artifacts):
        def generator_artifact(artifact: Artifact):
            yield AddArtifactRequest(metadata=MetaData(run_id=run_id,
                                                       type=artifact.artifact_type,
                                                       name=artifact.name))

            chunk_size = 1024
            for chunk in track(chunk_bytes(artifact.content, chunk_size), description='Uploading...'):
                yield AddArtifactRequest(content=Content(content=chunk))

        console.print(f'📤 Uploading {len(artifacts)} run artifacts', style='bold')

        for artifact in artifacts:
            try:
                console.print(f'📤 Uploading artifact <{artifact.name}> to mlsolid...', style='bold')
                self.stub.AddArtifact(generator_artifact(artifact))
                console.print(f'📥 Artifact <{artifact.name}> uploaded to mlsolid ✅', style='bold')
            except grpc.RpcError as e:
                raise self._handle_grpc_error(e)

    def create_model_registry(self, id: str):
        try:
            resp: CreateModelRegistryResponse = self.stub.CreateModelRegistry(CreateModelRegistryRequest(name=id))

            if resp.created:
                console.print(f'📑 ModelRegistry <{id}> created successfully')
            else:
                console.print('⚠️ Could not create ModelRegistry')

            return resp.created
        except grpc.RpcError as e:
            raise self._handle_grpc_error(e)
    
    def model_registry(self, id: str) -> ModelRegistry:
        try:
            resp: ModelRegistryResponse = self.stub.ModelRegistry(ModelRegistryRequest(name=id))

            return ModelRegistry(id=resp.name,
                                 entries=resp.model_entries,
                                 tags=resp.tags)
        except grpc.RpcError as e:
            raise self._handle_grpc_error(e)

    def add_model(self, id: str, run_id: str, artifact_id: str, tags: list[str]):
        try:
            resp: AddModelEntryResponse = self.stub.AddModelEntry(AddModelEntryRequest(name=id,
                                                                                       artifact_id=artifact_id,
                                                                                       run_id=run_id,
                                                                                       tags=tags))
            if resp.added:
                console.print(f'📑 Model <{artifact_id}> added to ModelRegistry <{id}> with tag <{tags}>', style='bold')

            return resp.added
        except grpc.RpcError as e:
            raise self._handle_grpc_error(e)

    def tagged_model(self, id: str, tag: str, path='./.mlsolid/models') -> str:
        try:
            resp: StreamTaggedModelResponse = self.stub.StreamTaggedModel(StreamTaggedModelRequest(name=id, tag=tag))

            metadata: MetaData = MetaData()
            content = bytearray()

            for response in track(resp, description='downloading...'):
                one = response.WhichOneof('response')
                if one == 'metadata':
                    metadata = response.metadata
                elif one == 'content':
                    content.extend(response.content.content)

            self._save_artifact(Artifact(name=metadata.name,
                                         artifact_type=ArtifactType.ModelArtifact.value,
                                         run_id=metadata.run_id,
                                         content=bytes(content)), path)
        except grpc.RpcError as e:
            raise self._handle_grpc_error(e)

    def _commit_run(self, run: RunManager):
        self.add_metrics(run.run_id, run.metrics)
        self.add_artifacts(run.run_id, run.artifacts)

    def _save_artifact(self, artifact: Artifact, path: str):
        path_obj = pathlib.Path(path)

        if path_obj.is_file():
            raise Exception(f'path <{path}> provided is a file')

        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)

        full_path = path_obj / artifact.name

        with open(full_path, 'bw') as f:
            f.write(artifact.content)

        console.print(f'📁 Artifact <{artifact.name}> saved to `{full_path}`', style='bold')

    def _handle_grpc_error(self, exception: grpc.RpcError) -> Exception:
        details = exception.details() # type: ignore

        match exception.code(): #type: ignore
            case grpc.StatusCode.INVALID_ARGUMENT:
                return BadRequest(details)
            case grpc.StatusCode.INTERNAL:
                return InternalError(details)
            case grpc.StatusCode.ALREADY_EXISTS:
                return BadRequest(details)
            case grpc.StatusCode.NOT_FOUND:
                return NotFound(details)

        return InternalError(details)

