import time
from collections import deque

class FatigueScorer:
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds

        # Rolling buffers storing (timestamp, value)
        self.eye_states = deque()       # 1 for closed, 0 for open
        self.distraction_states = deque() # 1 for distracted, 0 for attentive
        self.yawn_timestamps = deque()  # timestamps of individual yawns

        self.last_yawn_active = False

    def _purge_old_entries(self, now):
        cutoff = now - self.window_seconds
        while self.eye_states and self.eye_states[0][0] < cutoff:
            self.eye_states.popleft()
        while self.distraction_states and self.distraction_states[0][0] < cutoff:
            self.distraction_states.popleft()
        while self.yawn_timestamps and self.yawn_timestamps[0] < cutoff:
            self.yawn_timestamps.popleft()

    def update(self, is_eyes_closed, is_yawning, is_distracted):
        now = time.time()

        # Record instantaneous states
        self.eye_states.append((now, 1 if is_eyes_closed else 0))
        self.distraction_states.append((now, 1 if is_distracted else 0))

        # Yawn edge detector (only count rising edge to prevent multi-counting single yawn)
        if is_yawning and not self.last_yawn_active:
            self.yawn_timestamps.append(now)
        self.last_yawn_active = is_yawning

        self._purge_old_entries(now)

        # 1. PERCLOS score (0 to 100)
        perclos = 0.0
        if self.eye_states:
            closed_samples = sum(val for _, val in self.eye_states)
            perclos = (closed_samples / len(self.eye_states)) * 100.0

        # 2. Yawn frequency penalty (each yawn in last 60s adds 15 points)
        yawn_count = len(self.yawn_timestamps)
        yawn_score = min(45.0, yawn_count * 15.0)

        # 3. Cumulative gaze / phone distraction penalty
        distraction_score = 0.0
        if self.distraction_states:
            distracted_samples = sum(val for _, val in self.distraction_states)
            distraction_ratio = distracted_samples / len(self.distraction_states)
            distraction_score = min(25.0, distraction_ratio * 50.0)

        # Total weighted fatigue index capped at 100
        fatigue_index = min(100.0, (perclos * 1.5) + yawn_score + distraction_score)

        return round(fatigue_index, 1), yawn_count, round(perclos, 1)