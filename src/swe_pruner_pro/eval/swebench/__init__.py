"""SWE-Bench (mini-swe-agent fork with pruner integration).

The actual agent harness is the upstream
`SWE-agent/mini-swe-agent <https://github.com/SWE-agent/mini-swe-agent>`_
project, vendored under ``mini-swe-agent/`` with two additions:

1. ``minisweagent.utils.pruner.PrunerClient`` — HTTP client for the
   pruner-server's ``/prune`` endpoint, called between turns.
2. ``<output_threshold>X</output_threshold>`` XML-tag parsing (in
   ``DefaultAgent._get_output_threshold``) that lets the agent itself
   pick prune aggressiveness on a per-action basis.

See ``mini-swe-agent/README.md`` for the upstream docs and
``../README.md`` (this directory) for run instructions.
"""
