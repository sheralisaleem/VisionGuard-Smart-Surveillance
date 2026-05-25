import math
import time

class BehaviorAnalyzer:
    def __init__(self, loiter_time_sec=5, run_distance_px=50):
        self.loiter_threshold = loiter_time_sec
        self.run_threshold = run_distance_px
        self.first_seen = {}

    def detect_loitering(self, track_id):
        if track_id not in self.first_seen:
            self.first_seen[track_id] = time.time()
            return False
        
        time_visible = time.time() - self.first_seen[track_id]
        if time_visible > self.loiter_threshold:
            return True
        return False

    def detect_running(self, track_history):
        if len(track_history) < 5:
            return False 
        
        p1 = track_history[-5]
        p2 = track_history[-1]
        
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if distance > self.run_threshold:
            return True
        return False

    def detect_crowding(self, current_person_count, crowd_threshold=3):
        if current_person_count >= crowd_threshold:
            return True
        return False

    def detect_fight(self, track_history, current_positions, distance_threshold=80):
        """
        Detects fights by checking if two people are extremely close AND moving erratically.
        current_positions: dictionary {track_id: (center_x, center_y)}
        """
        fighting_ids = set()
        track_ids = list(current_positions.keys())
        
        # Compare every person to every other person
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                id1 = track_ids[i]
                id2 = track_ids[j]
                
                p1 = current_positions[id1]
                p2 = current_positions[id2]
                
                # Calculate Euclidean distance between the two people
                dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                # If they are overlapping or very close
                if dist < distance_threshold:
                    # Check if BOTH are moving fast (simulating a struggle)
                    if self.detect_running(track_history[id1]) and self.detect_running(track_history[id2]):
                        fighting_ids.add(id1)
                        fighting_ids.add(id2)
                        
        return fighting_ids