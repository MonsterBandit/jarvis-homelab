"""
Adapters produce request specs for callable tools.

They do NOT execute tools.
"""
from .local_read import request_read_file, request_read_snippet
from .web_okd import request_web_search, request_web_open
