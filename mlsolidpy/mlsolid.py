from collections import defaultdict
from typing import Any, Iterator, Dict
from contextlib import contextmanager

import grpc
import randomname
from rich.console import Console
from mlsolid.v1 import mlsolid_pb2_grpc
from mlsolid.v1.mlsolid_pb2 import AddMetricsRequest, CreateRunRequest, ExperimentRequest, ExperimentResponse, ExperimentsRequest, RunRequest, RunResponse

from mlsolidpy.exceptions import BadRequest, NotFound, InternalError
from mlsolidpy.types import FloatMetric, Metric, Run, StrMetric, from_protobuf_metrics, to_protobuf_metrics


console = Console()

class RunManager:
    def __init__(self, run_id: str, exp_id: str) -> None:
        self.exp_id = exp_id
        self.run_id = run_id
        self._metrics = defaultdict(list)

    @property
    def metrics(self) -> list[Metric]:
        return [self._parse_metric(name, vals) for name, vals in self._metrics.items()]

    def log(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            self._metrics[k].append(v)

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
    def __init__(self, url: str) -> None:
        self.url = url

        self.channel = grpc.insecure_channel(self.url)
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

    def _commit_run(self, run: RunManager):
        self.add_metrics(run.run_id, run.metrics)

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

