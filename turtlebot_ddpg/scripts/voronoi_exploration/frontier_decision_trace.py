"""Record frontier assignments and outcomes for one simulation run."""

import json
import threading
from pathlib import Path


class FrontierDecisionTrace:
    def __init__(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open('x', encoding='utf-8', buffering=1)
        self._lock = threading.Lock()
        self._pending = {}
        self._next_id = 1
        self._write(dict(event='run', schema_version=1, **metadata))

    def _write(self, row):
        self._file.write(json.dumps(row, allow_nan=False) + '\n')

    def assign(self, robot, elapsed, area, **selection):
        with self._lock:
            if self._file.closed:
                return
            self._finish(robot, 'replaced', elapsed, area)
            attempt = self._next_id
            self._next_id += 1
            self._pending[robot] = attempt, area
            self._write(dict(event='assigned', attempt_id=attempt, robot=robot,
                             elapsed_sim_s=elapsed, team_area_m2=area, **selection))

    def _finish(self, robot, outcome, elapsed, area):
        pending = self._pending.pop(robot, None)
        if pending is None:
            return
        attempt, start_area = pending
        self._write(dict(event='outcome', attempt_id=attempt, robot=robot,
                         outcome=outcome, elapsed_sim_s=elapsed,
                         team_area_m2=area, team_area_gain_m2=area - start_area))

    def finish(self, robot, outcome, elapsed, area):
        with self._lock:
            if not self._file.closed:
                self._finish(robot, outcome, elapsed, area)

    def close(self, outcome, elapsed, area):
        with self._lock:
            if self._file.closed:
                return
            for robot in list(self._pending):
                self._finish(robot, outcome, elapsed, area)
            self._write(dict(event='run_end', reason=outcome,
                             elapsed_sim_s=elapsed, team_area_m2=area))
            self._file.close()
