class RiskService:
    """
    Rule-based forensic risk scoring from YOLO detections.
    High weight on weapons + person combinations and blood evidence.
    """

    def calculate(self, detections: list) -> dict:
        if not detections:
            return {
                "threat_level": "None",
                "severity": "Low",
                "probability": 0.05,
                "impact_score": 0,
                "overall_risk": 0,
                "risk_label": "Safe",
                "weapon_count": 0,
                "reasons": ["No objects of forensic interest detected"]
            }

        score = 0.0
        reasons = []
        has_weapon = False
        has_person = False
        has_blood = False
        weapon_count = 0

        for det in detections:
            label = det.get("class", "").lower()
            conf = float(det.get("confidence", 0))
            category = det.get("category", "Others")

            if category == "Weapons" or any(w in label for w in
                ["gun", "pistol", "rifle", "knife", "firearm", "weapon", "blade"]):
                has_weapon = True
                weapon_count += 1
                points = 45 * conf
                score += points
                reasons.append(f"Weapon: {det['class']} ({conf:.0%})")

            elif category == "Blood / Biological" or "blood" in label:
                has_blood = True
                score += 35 * conf
                reasons.append(f"Blood evidence: {det['class']} ({conf:.0%})")

            elif category == "Persons" or "person" in label:
                has_person = True
                score += 12 * conf

            elif category == "Vehicles":
                score += 6 * conf

        # Interaction bonuses (very important for crime scenes)
        if has_weapon and has_person:
            score += 30
            reasons.append("CRITICAL: Person + Weapon combination")
        if has_blood and has_weapon:
            score += 20
            reasons.append("Blood + Weapon present")
        if has_blood and has_person:
            score += 15
            reasons.append("Blood + Person present")

        overall = min(100, int(round(score)))

        if overall >= 75:
            threat, severity, label = "Critical", "High", "High Risk – Immediate Attention"
        elif overall >= 45:
            threat, severity, label = "Elevated", "Medium-High", "Medium-High Risk"
        elif overall >= 20:
            threat, severity, label = "Moderate", "Medium", "Medium Risk"
        else:
            threat, severity, label = "Low", "Low", "Low Risk"

        probability = round(min(0.98, overall / 100 * 0.9 + 0.05), 2)

        person_count = sum(
            1 for d in detections
            if d.get("category") == "Persons" or "person" in str(d.get("class", "")).lower()
        )

        return {
            "threat_level": threat,
            "severity": severity,
            "probability": probability,
            "impact_score": overall,
            "overall_risk": overall,
            "risk_label": label,
            "weapon_count": weapon_count,
            "person_count": person_count,
            "has_blood": has_blood,
            "has_person": has_person,
            "reasons": reasons or ["Minor / non-critical objects detected"]
        }
