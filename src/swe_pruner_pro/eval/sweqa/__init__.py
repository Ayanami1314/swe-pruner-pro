"""SWE-QA / SWE-QA-Pro evaluation harness.

Two scripts:

* ``agent_eval.py`` — bash-agent loop that explores cloned code
  repositories and answers questions. Optional pruning via an external
  pruner-server (``--pruner-url``) or a client-side baseline
  (``--baseline``).
* ``judge.py``     — GPT-5.4-mini 5-dim LLM-as-judge for produced
  ``*_answers.jsonl`` files.

Run ``setup`` once to clone the three SWE-QA repos. For SWE-QA-Pro run
``prepare-pro`` (downloads ``TIGER-Lab/SWE-QA-Pro-Bench`` from HF) +
``setup --variant sweqa-pro``.
"""
