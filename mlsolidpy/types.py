import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from mlsolid.v1.mlsolid_pb2 import Metric as p_Metric, ModelEntry, Val


class Metric(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def vals(self) -> list[Any]:
        ...

    def to_protobuf(self) -> p_Metric:
        ...

@dataclass
class FloatMetric:
    name: str
    vals: list[float]

    def to_protobuf(self) -> p_Metric:
        vals = [Val(double=val) for val in self.vals]
        return p_Metric(name=self.name, vals=vals)

@dataclass
class IntMetric:
    name: str
    vals: list[int]

    def to_protobuf(self) -> p_Metric:
        vals = [Val(int=val) for val in self.vals]
        return p_Metric(name=self.name, vals=vals)

@dataclass
class StrMetric:
    name: str
    vals: list[str]

    def to_protobuf(self) -> p_Metric:
        vals = [Val(str=val) for val in self.vals]
        return p_Metric(name=self.name, vals=vals)

@dataclass
class Run:
    id: str
    timestamp: datetime.datetime
    exp_id: str
    metrics: list[Metric]

@dataclass
class ModelRegistry:
    id: str
    entries: list[ModelEntry]
    tags : dict[str, list[str]]

@dataclass
class ModelEntry:
    url: str
    tags: str

@dataclass
class Artifact:
    name: str
    artifact_type: str
    run_id: str
    content: bytes

class ArtifactType(Enum):
    ModelArtifact = "content-type/model"
    PlainTextArtifact = "content-type/text"

def to_protobuf_metrics(metrics: list[Metric]) -> list[p_Metric]:
    return [metric.to_protobuf() for metric in metrics]

def from_protobuf_metrics(metrics: list[p_Metric]) -> list[Metric]:
    return [from_protobuf_metric(m) for m in metrics]

def from_protobuf_metric(metric: p_Metric) -> Metric:
    if len(metric.vals) == 0:
        return StrMetric(name=metric.name, vals=[])

    oneof = getattr(metric.vals[0], metric.vals[0].WhichOneof('val'))

    out_metric: Metric

    if type(oneof) is float:
        out_metric = FloatMetric(name=metric.name, vals=[])
    elif type(oneof) is int:
        out_metric = IntMetric(name=metric.name, vals=[])
    else:
        out_metric = StrMetric(name=metric.name, vals=[])

    for val in metric.vals:
        out_metric.vals.append(getattr(val, val.WhichOneof('val')))

    return out_metric

def chunk_bytes(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]
