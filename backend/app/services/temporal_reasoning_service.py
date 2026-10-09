from itertools import combinations


class TemporalReasoningService:
    FIREARM_CLASSES = {"gun", "pistol", "rifle"}
    WEAPON_CLASSES = FIREARM_CLASSES | {"weapon", "knife", "hammer"}

    @classmethod
    def aggregate_evidence(cls, frame_results, max_gap, min_confirmations):
        observations = {}
        for frame in frame_results:
            for detection in frame["detections"]:
                track_id = detection.get("track_id")
                if track_id is None:
                    continue
                key = (detection["class"], track_id)
                observations.setdefault(key, []).append({
                    "timestamp": detection["timestamp"],
                    "frame_number": detection["frame_number"],
                    "confidence": detection["confidence"],
                })

        tracks = []
        for (object_class, track_id), samples in observations.items():
            samples.sort(key=lambda sample: sample["timestamp"])
            confirmed_segments = [
                segment for segment in cls._split_temporal_samples(samples, max_gap)
                if len(segment) >= min_confirmations
            ]
            confirmed_samples = [sample for segment in confirmed_segments for sample in segment]
            if not confirmed_samples:
                continue
            confidences = [sample["confidence"] for sample in confirmed_samples]
            tracks.append({
                "class": object_class,
                "track_id": track_id,
                "first_seen": confirmed_samples[0]["timestamp"],
                "last_seen": confirmed_samples[-1]["timestamp"],
                "samples": len(confirmed_samples),
                "frames_detected": len(confirmed_samples),
                "mean_confidence": round(sum(confidences) / len(confidences), 3),
                "maximum_confidence": round(max(confidences), 3),
                "confidence": round(max(confidences), 3),
                "segments": [
                    {
                        "first_seen": segment[0]["timestamp"],
                        "last_seen": segment[-1]["timestamp"],
                        "frame_start": segment[0]["frame_number"],
                        "frame_end": segment[-1]["frame_number"],
                        "samples": len(segment),
                    }
                    for segment in confirmed_segments
                ],
            })

        for evidence in tracks:
            evidence["label"] = f"{evidence['class'].title()} #{evidence['track_id']}"

        return sorted(tracks, key=lambda item: (item["first_seen"], item["class"], item["track_id"]))

    @classmethod
    def reason_events(
        cls,
        frame_results,
        evidence,
        frame_width,
        frame_height,
        proximity_ratio,
        approach_ratio,
        event_gap,
        min_confirmations,
    ):
        if not evidence:
            return []

        confirmed_tracks = {
            (item["class"], item["track_id"]): item for item in evidence
        }
        person_tracks = {
            item["track_id"]: item
            for item in evidence
            if item["class"] == "person"
        }
        weapon_tracks = [
            item for item in evidence if item["class"] in cls.WEAPON_CLASSES
        ]
        diagonal = max(1.0, (frame_width ** 2 + frame_height ** 2) ** 0.5)
        pair_observations = {}
        weapon_person_cooccurrences = {}
        weapon_sample_counts = {}

        for frame in frame_results:
            detections = [
                detection
                for detection in frame["detections"]
                if (detection["class"], detection.get("track_id")) in confirmed_tracks
            ]
            people = [detection for detection in detections if detection["class"] == "person"]
            weapons = [detection for detection in detections if detection["class"] in cls.WEAPON_CLASSES]

            for first, second in combinations(people, 2):
                pair_key = tuple(sorted((first["track_id"], second["track_id"])))
                center_distance = cls._center_distance(first["bbox"], second["bbox"]) / diagonal
                box_distance = cls._box_distance(first["bbox"], second["bbox"]) / diagonal
                pair_observations.setdefault(pair_key, []).append({
                    "timestamp": frame["timestamp_sec"],
                    "frame_number": frame["frame_number"],
                    "center_distance": center_distance,
                    "close": box_distance <= proximity_ratio,
                })

            for weapon in weapons:
                weapon_key = (weapon["class"], weapon["track_id"])
                weapon_sample_counts[weapon_key] = weapon_sample_counts.get(weapon_key, 0) + 1
                for person in people:
                    distance = cls._box_distance(weapon["bbox"], person["bbox"]) / diagonal
                    if distance <= proximity_ratio:
                        pair_key = (weapon["class"], weapon["track_id"], person["track_id"])
                        weapon_person_cooccurrences.setdefault(pair_key, []).append(
                            frame["timestamp_sec"]
                        )

        person_pairs = []
        for pair_key, samples in pair_observations.items():
            samples.sort(key=lambda sample: sample["timestamp"])
            for episode in cls._split_temporal_samples(samples, event_gap):
                close_samples = [sample for sample in episode if sample["close"]]
                is_approaching = (
                    len(episode) >= min_confirmations
                    and episode[-1]["center_distance"] <= proximity_ratio * 2
                    and episode[0]["center_distance"] - episode[-1]["center_distance"] >= approach_ratio
                )
                if len(close_samples) >= min_confirmations or is_approaching:
                    person_pairs.append((pair_key, {
                        "samples": episode,
                        "close_samples": close_samples,
                        "approaching": is_approaching,
                    }))

        events = []
        for weapon in weapon_tracks:
            weapon_key = (weapon["class"], weapon["track_id"])
            nearby_persons = {
                person_id: timestamps
                for (object_class, track_id, person_id), timestamps in weapon_person_cooccurrences.items()
                if object_class == weapon["class"] and track_id == weapon["track_id"]
                and len(timestamps) >= min_confirmations
            }
            associated_person_id = max(nearby_persons, key=lambda person_id: len(nearby_persons[person_id]), default=None)
            if associated_person_id is not None:
                association_samples = len(nearby_persons[associated_person_id])
                association_score = association_samples / max(1, weapon_sample_counts.get(weapon_key, 0))
                weapon["associated_person"] = {
                    "track_id": associated_person_id,
                    "label": person_tracks[associated_person_id]["label"],
                }
                weapon["association_samples"] = association_samples
                weapon["association_score"] = round(min(1.0, association_score), 3)
            else:
                weapon["associated_person"] = None
                weapon["association_samples"] = 0
                weapon["association_score"] = 0.0
            confrontation = None
            for pair_key, pair_data in person_pairs:
                if not nearby_persons or not set(pair_key).intersection(nearby_persons):
                    continue
                close_times = [sample["timestamp"] for sample in pair_data["close_samples"]]
                if not close_times and pair_data["approaching"]:
                    close_times = [sample["timestamp"] for sample in pair_data["samples"]]
                near_times = sorted(
                    timestamp
                    for person_id in pair_key
                    for timestamp in nearby_persons.get(person_id, [])
                )
                common_times = cls._matching_times(close_times, near_times, max_delta=event_gap)
                if len(common_times) >= min_confirmations:
                    candidate = (len(common_times), pair_key, pair_data, common_times)
                    if confrontation is None or candidate[0] > confrontation[0]:
                        confrontation = candidate

            if confrontation:
                _, person_pair, pair_data, event_times = confrontation
                first_person = person_tracks[person_pair[0]]
                second_person = person_tracks[person_pair[1]]
                nearby_id = next(
                    person_id for person_id in person_pair if person_id in nearby_persons
                )
                weapon_label = weapon["label"]
                nearby_label = person_tracks[nearby_id]["label"]
                if weapon["class"] == "knife":
                    event_type = "possible_knife_related_threat"
                    interpretation = "Possible knife-related threat or confrontation"
                elif weapon["class"] in cls.FIREARM_CLASSES:
                    event_type = "possible_firearm_related_confrontation"
                    interpretation = "Possible firearm-related confrontation"
                else:
                    event_type = "possible_weapon_related_confrontation"
                    interpretation = "Possible weapon-related confrontation"

                observations = [
                    f"{first_person['label']} and {second_person['label']} were both detected in {len(event_times)} temporally aligned sampled frames.",
                    f"{weapon_label} was detected near {nearby_label} in multiple sampled frames.",
                ]
                if pair_data["approaching"]:
                    observations.append(
                        f"The bounding-box centers for {first_person['label']} and {second_person['label']} moved closer across sampled frames."
                    )
                else:
                    observations.append(
                        f"The persons' bounding boxes remained in close proximity across sampled frames."
                    )
                explanation = (
                    f"{interpretation} is a cautious interpretation of the listed observations. "
                    "Object detections cannot establish possession, intent, physical contact, or assault."
                )
                persons = [
                    {"track_id": first_person["track_id"], "label": first_person["label"]},
                    {"track_id": second_person["track_id"], "label": second_person["label"]},
                ]
                start_time = event_times[0]
                end_time = event_times[-1]
            else:
                if weapon["class"] in cls.FIREARM_CLASSES:
                    event_type = "firearm_detected"
                    interpretation = "Firearm detected; firing cannot be confirmed from object detection alone"
                elif weapon["class"] == "knife":
                    event_type = "knife_detected"
                    interpretation = "Knife detected; no confrontation was established from the sampled boxes"
                else:
                    event_type = "weapon_detected"
                    interpretation = "Weapon object detected; no confrontation was established from the sampled boxes"
                nearest_person = min(
                    (person_id for person_id in person_tracks if person_id in nearby_persons),
                    key=lambda person_id: -len(nearby_persons[person_id]),
                    default=None,
                )
                persons = ([{
                    "track_id": nearest_person,
                    "label": person_tracks[nearest_person]["label"],
                }] if nearest_person is not None else [])
                subject = persons[0]["label"] if persons else "a person"
                observations = [f"{weapon['label']} detected in {weapon['frames_detected']} sampled frames."]
                if persons:
                    observations.append(f"{weapon['label']} was detected near {subject} in multiple sampled frames.")
                explanation = (
                    f"{interpretation}. This does not establish whether the object was held, fired, or used."
                )
                start_time = weapon["first_seen"]
                end_time = weapon["last_seen"]

            mean_confidence = weapon["mean_confidence"]
            support = min(0.15, max(0, weapon["frames_detected"] - min_confirmations) * 0.03)
            relationship_support = 0.08 if confrontation else 0.0
            heuristic_confidence = min(0.79, max(0.5, 0.5 + 0.25 * max(0, mean_confidence - 0.4) / 0.6 + support + relationship_support))
            events.append({
                "start_time": start_time,
                "end_time": end_time,
                "event_type": event_type,
                "interpretation": interpretation,
                "confidence": round(heuristic_confidence, 2),
                "confidence_label": "Medium",
                "confidence_basis": "Heuristic evidence-strength score, not a calibrated probability.",
                "track_id": weapon["track_id"],
                "persons": persons,
                "evidence": sorted({weapon["class"], *("person" for _ in persons)}),
                "observations": observations,
                "explanation": explanation,
                "robbery_assessment": "Not classified: these object classes do not provide reliable evidence of object transfer or ownership.",
            })

        events.extend(cls._multiple_person_events(
            frame_results, person_tracks, event_gap, min_confirmations
        ))
        for pair_key, pair_data in person_pairs:
            first_person = person_tracks[pair_key[0]]
            second_person = person_tracks[pair_key[1]]
            samples = pair_data["close_samples"] or pair_data["samples"]
            approaching = pair_data["approaching"]
            events.append({
                "start_time": samples[0]["timestamp"],
                "end_time": samples[-1]["timestamp"],
                "event_type": "person_approach_observed" if approaching else "person_proximity_observed",
                "interpretation": "Person bounding boxes moved closer" if approaching else "Persons in close proximity",
                "confidence": round((first_person["mean_confidence"] + second_person["mean_confidence"]) / 2, 3),
                "confidence_label": "Detection confidence",
                "confidence_basis": "Mean YOLO confidence for the two person tracks; not an action-recognition score.",
                "track_id": None,
                "persons": [
                    {"track_id": first_person["track_id"], "label": first_person["label"]},
                    {"track_id": second_person["track_id"], "label": second_person["label"]},
                ],
                "evidence": ["person"],
                "observations": [
                    f"{first_person['label']} and {second_person['label']} were both detected across {len(samples)} sampled frames.",
                    "Bounding-box centers moved closer in the image." if approaching else "The bounding boxes remained in close proximity.",
                ],
                "explanation": "This is a spatial observation only; it does not establish social interaction, following, confrontation, or intent.",
                "robbery_assessment": "Not classified: these object classes do not provide reliable evidence of object transfer or ownership.",
            })

        return sorted(events, key=lambda event: (event["start_time"], event["event_type"]))

    @staticmethod
    def _multiple_person_events(frame_results, person_tracks, max_gap, min_confirmations):
        events = []
        current = None
        for frame in frame_results:
            detections_by_track = {
                detection["track_id"]: detection
                for detection in frame["detections"]
                if detection["class"] == "person" and detection.get("track_id") in person_tracks
            }
            people = list(detections_by_track.values())
            if len(people) < 2:
                if current and frame["timestamp_sec"] - current["end_time"] > max_gap:
                    TemporalReasoningService._finish_person_presence(events, current, min_confirmations)
                    current = None
                continue

            timestamp = frame["timestamp_sec"]
            if current is None or timestamp - current["end_time"] > max_gap:
                if current:
                    TemporalReasoningService._finish_person_presence(events, current, min_confirmations)
                current = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "frame_start": frame["frame_number"],
                    "frame_end": frame["frame_number"],
                    "track_ids": set(),
                    "confidences": [],
                    "frames": 0,
                }
            current["end_time"] = timestamp
            current["frame_end"] = frame["frame_number"]
            current["track_ids"].update(detection["track_id"] for detection in people)
            current["confidences"].extend(detection["confidence"] for detection in people)
            current["frames"] += 1

        if current:
            TemporalReasoningService._finish_person_presence(events, current, min_confirmations)
        return events

    @staticmethod
    def _finish_person_presence(events, current, min_confirmations):
        if current["frames"] < min_confirmations:
            return
        persons = sorted(current["track_ids"])
        events.append({
            "start_time": current["start_time"],
            "end_time": current["end_time"],
            "event_type": "multiple_persons_detected",
            "interpretation": "Multiple persons detected",
            "confidence": round(sum(current["confidences"]) / len(current["confidences"]), 3),
            "confidence_label": "Detection confidence",
            "confidence_basis": "Mean YOLO confidence across the confirmed person detections.",
            "track_id": None,
            "persons": [
                {"track_id": track_id, "label": f"Person #{track_id}"}
                for track_id in persons
            ],
            "evidence": ["person"],
            "observations": [
                f"{len(persons)} person tracks were simultaneously visible in {current['frames']} sampled frames."
            ],
            "explanation": "Co-presence only; the detections do not demonstrate interaction or criminal activity.",
            "robbery_assessment": "Not classified: these object classes do not provide reliable evidence of object transfer or ownership.",
        })

    @staticmethod
    def _split_temporal_samples(samples, max_gap):
        episodes = []
        current = []
        for sample in samples:
            if current and sample["timestamp"] - current[-1]["timestamp"] > max_gap:
                episodes.append(current)
                current = []
            current.append(sample)
        if current:
            episodes.append(current)
        return episodes

    @staticmethod
    def _matching_times(first_times, second_times, max_delta):
        matched = []
        for first_time in first_times:
            if any(abs(first_time - second_time) <= max_delta for second_time in second_times):
                matched.append(first_time)
        return matched

    @staticmethod
    def _center_distance(first, second):
        first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
        second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
        return ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5

    @staticmethod
    def _box_distance(first, second):
        horizontal_gap = max(first[0] - second[2], second[0] - first[2], 0)
        vertical_gap = max(first[1] - second[3], second[1] - first[3], 0)
        return (horizontal_gap ** 2 + vertical_gap ** 2) ** 0.5
