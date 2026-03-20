"""archivist:// URI scheme — addressable references to memories, entities, and namespaces.

Format:
  archivist://{namespace}/{resource_type}/{id}[?params]

Resource types:
  memory/{point_id}     — a Qdrant memory point
  entity/{entity_id}    — a knowledge graph entity
  namespace/{name}      — browsable namespace root
  skill/{skill_id}      — a registered skill

Examples:
  archivist://agents-nova/memory/abc123-def456
  archivist://shared/entity/42
  archivist://agents-nova/skill/web_search
"""

import re
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlencode, parse_qs


@dataclass
class ArchivistURI:
    namespace: str
    resource_type: str
    resource_id: str
    params: dict

    def __str__(self) -> str:
        base = f"archivist://{quote(self.namespace, safe='')}/{self.resource_type}/{quote(str(self.resource_id), safe='')}"
        if self.params:
            base += "?" + urlencode(self.params, doseq=True)
        return base

    @property
    def is_memory(self) -> bool:
        return self.resource_type == "memory"

    @property
    def is_entity(self) -> bool:
        return self.resource_type == "entity"

    @property
    def is_namespace(self) -> bool:
        return self.resource_type == "namespace"

    @property
    def is_skill(self) -> bool:
        return self.resource_type == "skill"


_URI_PATTERN = re.compile(
    r"^archivist://(?P<namespace>[^/]+)/(?P<resource_type>memory|entity|namespace|skill)/(?P<resource_id>[^?]+)(?:\?(?P<params>.+))?$"
)


def parse_uri(uri: str) -> ArchivistURI | None:
    """Parse an archivist:// URI string. Returns None if malformed."""
    m = _URI_PATTERN.match(uri.strip())
    if not m:
        return None
    params = {}
    if m.group("params"):
        params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(m.group("params")).items()}
    return ArchivistURI(
        namespace=unquote(m.group("namespace")),
        resource_type=m.group("resource_type"),
        resource_id=unquote(m.group("resource_id")),
        params=params,
    )


def memory_uri(namespace: str, point_id: str, **params) -> str:
    return str(ArchivistURI(namespace=namespace, resource_type="memory", resource_id=point_id, params=params))


def entity_uri(namespace: str, entity_id: str | int, **params) -> str:
    return str(ArchivistURI(namespace=namespace, resource_type="entity", resource_id=str(entity_id), params=params))


def namespace_uri(namespace: str, **params) -> str:
    return str(ArchivistURI(namespace=namespace, resource_type="namespace", resource_id=namespace, params=params))


def skill_uri(namespace: str, skill_id: str, **params) -> str:
    return str(ArchivistURI(namespace=namespace, resource_type="skill", resource_id=skill_id, params=params))
