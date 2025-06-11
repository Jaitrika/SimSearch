from uuid import uuid4

class SimpleNode:
    def __init__(self, text, node_id=None, metadata=None, ref_doc_id=None):
        self.text = text
        self.node_id = node_id or str(uuid4())
        self.metadata = metadata or {}
        self.ref_doc_id = ref_doc_id

    def __repr__(self):
        return f"SimpleNode(text={self.text[:30]!r}, node_id={self.node_id})"
