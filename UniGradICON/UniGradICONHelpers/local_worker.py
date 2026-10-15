"""Isolated local inference process used by the interactive registration task."""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from UniGradICONHelpers import registration


def run(manifest, report):
  arguments = dict(manifest["arguments"])

  def progress(total, step, elapsed=None, remaining=None):
    report({"phase": "Registering", "total": total, "step": step,
            "elapsed": elapsed, "remaining": remaining})

  result = registration.register_files(
    **arguments, call_back=progress, phase_callback=lambda phase: report({"phase": phase}))
  report({"state": "done", "transform": result[0], "warped": result[1]})


def main(path):
  with open(path) as file:
    manifest = json.load(file)
  directory = os.path.dirname(path)

  def report(status):
    temporary = os.path.join(directory, "status.tmp")
    with open(temporary, "w") as file:
      json.dump(status, file)
    os.replace(temporary, os.path.join(directory, "status.json"))

  def watch():
    # Each job owns its process, so cancellation can also stop a long forward pass
    # with zero IO steps, without touching another registration's cached model.
    while True:
      if (os.path.exists(os.path.join(directory, "cancel"))
          or not registration.process_exists(manifest["parent_pid"])):
        os._exit(2)
      time.sleep(0.1)

  threading.Thread(target=watch, daemon=True).start()
  try:
    run(manifest, report)
  except Exception as error:
    report({"state": "failed", "error": f"{type(error).__name__}: {error}"})
    raise


if __name__ == "__main__":
  main(sys.argv[1])
