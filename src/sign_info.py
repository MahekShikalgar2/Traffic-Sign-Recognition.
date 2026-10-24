"""
sign_info.py — Static metadata for all 43 GTSRB traffic sign classes.

Keys are class IDs 0–42 (matching model output).
Note: signs.py uses 1-indexed keys (1–43), so class_id + 1 maps to that dict.

Each entry contains:
    name         : Human-readable sign name (matches signs.classes[id+1])
    category     : One of: Speed Limits, Prohibitory, Mandatory, Warning, Other
    meaning      : 1-sentence plain English description of the sign.
    driving_tip  : 1-sentence advice for a driver seeing this sign.
"""

SIGN_INFO: dict[int, dict] = {
    0: {
        "name": "Speed limit (20km/h)",
        "category": "Speed Limits",
        "meaning": "Maximum permitted speed is 20 km/h in this zone.",
        "driving_tip": "Reduce your speed to 20 km/h or below immediately.",
    },
    1: {
        "name": "Speed limit (30km/h)",
        "category": "Speed Limits",
        "meaning": "Maximum permitted speed is 30 km/h in this zone.",
        "driving_tip": "Slow down to 30 km/h or below, typically used in residential areas.",
    },
    2: {
        "name": "Speed limit (50km/h)",
        "category": "Speed Limits",
        "meaning": "Maximum permitted speed is 50 km/h in this zone.",
        "driving_tip": "Maintain speed at or below 50 km/h through this area.",
    },
    3: {
        "name": "Speed limit (60km/h)",
        "category": "Speed Limits",
        "meaning": "Maximum permitted speed is 60 km/h in this zone.",
        "driving_tip": "Do not exceed 60 km/h until the limit changes.",
    },
    4: {
        "name": "Speed limit (70km/h)",
        "category": "Speed Limits",
        "meaning": "Maximum permitted speed is 70 km/h in this zone.",
        "driving_tip": "Keep your speed at or below 70 km/h in this section.",
    },
    5: {
        "name": "Speed limit (80km/h)",
        "category": "Speed Limits",
        "meaning": "Maximum permitted speed is 80 km/h in this zone.",
        "driving_tip": "Do not exceed 80 km/h until you see a new speed sign.",
    },
    6: {
        "name": "End of speed limit (80km/h)",
        "category": "Speed Limits",
        "meaning": "The 80 km/h speed restriction ends here.",
        "driving_tip": "The previous 80 km/h limit no longer applies; default national limits resume.",
    },
    7: {
        "name": "Speed limit (100km/h)",
        "category": "Speed Limits",
        "meaning": "Maximum permitted speed is 100 km/h in this zone.",
        "driving_tip": "Do not exceed 100 km/h; remain alert for upcoming lower-limit signs.",
    },
    8: {
        "name": "Speed limit (120km/h)",
        "category": "Speed Limits",
        "meaning": "Maximum permitted speed is 120 km/h in this zone.",
        "driving_tip": "This is typically a motorway or expressway; maintain steady speed below 120 km/h.",
    },
    9: {
        "name": "No passing",
        "category": "Prohibitory",
        "meaning": "Overtaking other vehicles is prohibited in this zone.",
        "driving_tip": "Stay in your lane and do not attempt to pass any vehicle ahead.",
    },
    10: {
        "name": "No passing veh over 3.5 tons",
        "category": "Prohibitory",
        "meaning": "Vehicles heavier than 3.5 tonnes are not allowed to overtake here.",
        "driving_tip": "If driving a heavy vehicle, do not overtake; light vehicles may pass with caution.",
    },
    11: {
        "name": "Right-of-way at intersection",
        "category": "Warning",
        "meaning": "You have the right of way at the next intersection.",
        "driving_tip": "Proceed through the intersection but remain watchful for drivers who may not yield.",
    },
    12: {
        "name": "Priority road",
        "category": "Mandatory",
        "meaning": "You are on a priority road and have right of way over joining traffic.",
        "driving_tip": "Maintain your speed but stay alert for vehicles entering from side roads.",
    },
    13: {
        "name": "Yield",
        "category": "Prohibitory",
        "meaning": "You must give way to all traffic on the road you are joining.",
        "driving_tip": "Slow down and yield to oncoming traffic before proceeding.",
    },
    14: {
        "name": "Stop",
        "category": "Prohibitory",
        "meaning": "You must come to a complete stop before proceeding.",
        "driving_tip": "Bring the vehicle to a full stop, check for crossing traffic, then proceed safely.",
    },
    15: {
        "name": "No vehicles",
        "category": "Prohibitory",
        "meaning": "No vehicles of any kind are permitted beyond this sign.",
        "driving_tip": "Do not enter this road; find an alternative route.",
    },
    16: {
        "name": "Veh > 3.5 tons prohibited",
        "category": "Prohibitory",
        "meaning": "Vehicles exceeding 3.5 tonnes are not permitted to proceed.",
        "driving_tip": "Heavy vehicle drivers must turn around and use a designated alternative route.",
    },
    17: {
        "name": "No entry",
        "category": "Prohibitory",
        "meaning": "Entry to this road is strictly prohibited for all vehicles.",
        "driving_tip": "Do not enter — this is typically a one-way road in the opposite direction.",
    },
    18: {
        "name": "General caution",
        "category": "Warning",
        "meaning": "A hazard or dangerous condition lies ahead.",
        "driving_tip": "Reduce speed and be prepared to react to an unspecified hazard ahead.",
    },
    19: {
        "name": "Dangerous curve left",
        "category": "Warning",
        "meaning": "There is a sharp or dangerous curve to the left ahead.",
        "driving_tip": "Slow down before the curve and steer carefully to the left.",
    },
    20: {
        "name": "Dangerous curve right",
        "category": "Warning",
        "meaning": "There is a sharp or dangerous curve to the right ahead.",
        "driving_tip": "Reduce speed before the curve and steer carefully to the right.",
    },
    21: {
        "name": "Double curve",
        "category": "Warning",
        "meaning": "Two consecutive sharp curves lie ahead.",
        "driving_tip": "Slow down and be ready to navigate two bends in quick succession.",
    },
    22: {
        "name": "Bumpy road",
        "category": "Warning",
        "meaning": "The road surface ahead is uneven or contains bumps.",
        "driving_tip": "Reduce speed to avoid vehicle damage and loss of control on the rough surface.",
    },
    23: {
        "name": "Slippery road",
        "category": "Warning",
        "meaning": "The road surface ahead may be slippery, especially when wet.",
        "driving_tip": "Reduce speed and avoid sudden braking or sharp steering on the slippery surface.",
    },
    24: {
        "name": "Road narrows on the right",
        "category": "Warning",
        "meaning": "The road becomes narrower on the right side ahead.",
        "driving_tip": "Move toward the centre carefully and watch for oncoming traffic in the narrow section.",
    },
    25: {
        "name": "Road work",
        "category": "Warning",
        "meaning": "Road works or construction activity is present ahead.",
        "driving_tip": "Slow down, follow any temporary signs, and watch out for workers and machinery.",
    },
    26: {
        "name": "Traffic signals",
        "category": "Warning",
        "meaning": "Traffic lights are located ahead.",
        "driving_tip": "Be prepared to stop if the signal is red as you approach.",
    },
    27: {
        "name": "Pedestrians",
        "category": "Warning",
        "meaning": "Pedestrians may be crossing or walking alongside the road ahead.",
        "driving_tip": "Slow down and yield to any pedestrians in or near the roadway.",
    },
    28: {
        "name": "Children crossing",
        "category": "Warning",
        "meaning": "Children may be crossing the road, often near a school.",
        "driving_tip": "Reduce speed significantly and be ready to stop for children crossing.",
    },
    29: {
        "name": "Bicycles crossing",
        "category": "Warning",
        "meaning": "Cyclists may be crossing or entering the road ahead.",
        "driving_tip": "Slow down and watch carefully for cyclists at the crossing point.",
    },
    30: {
        "name": "Beware of ice/snow",
        "category": "Warning",
        "meaning": "Ice or snow on the road surface creates a hazardous condition.",
        "driving_tip": "Reduce speed greatly and use gentle inputs to avoid skidding on icy or snowy road.",
    },
    31: {
        "name": "Wild animals crossing",
        "category": "Warning",
        "meaning": "Wild animals may cross the road unexpectedly in this area.",
        "driving_tip": "Stay alert especially at dawn and dusk, and slow down when animals are near the road.",
    },
    32: {
        "name": "End speed + passing limits",
        "category": "Other",
        "meaning": "All previously imposed speed and overtaking restrictions end here.",
        "driving_tip": "Normal national speed and passing rules resume beyond this sign.",
    },
    33: {
        "name": "Turn right ahead",
        "category": "Mandatory",
        "meaning": "You must turn right at the upcoming junction.",
        "driving_tip": "Position your vehicle in the right lane and prepare to turn right.",
    },
    34: {
        "name": "Turn left ahead",
        "category": "Mandatory",
        "meaning": "You must turn left at the upcoming junction.",
        "driving_tip": "Move into the left lane and signal your intention to turn left.",
    },
    35: {
        "name": "Ahead only",
        "category": "Mandatory",
        "meaning": "You must continue straight ahead and may not turn.",
        "driving_tip": "Proceed straight through the intersection without turning.",
    },
    36: {
        "name": "Go straight or right",
        "category": "Mandatory",
        "meaning": "You may only go straight ahead or turn right at the junction.",
        "driving_tip": "Turning left is not permitted; choose to go straight or right.",
    },
    37: {
        "name": "Go straight or left",
        "category": "Mandatory",
        "meaning": "You may only go straight ahead or turn left at the junction.",
        "driving_tip": "Turning right is not permitted; choose to go straight or turn left.",
    },
    38: {
        "name": "Keep right",
        "category": "Mandatory",
        "meaning": "You must keep to the right side of the obstacle or divider.",
        "driving_tip": "Stay on the right side of the road feature indicated by this sign.",
    },
    39: {
        "name": "Keep left",
        "category": "Mandatory",
        "meaning": "You must keep to the left side of the obstacle or divider.",
        "driving_tip": "Stay on the left side of the road feature indicated by this sign.",
    },
    40: {
        "name": "Roundabout mandatory",
        "category": "Mandatory",
        "meaning": "You must follow the roundabout traffic flow in the direction indicated.",
        "driving_tip": "Enter the roundabout, travel anti-clockwise (or as indicated), and exit at your desired road.",
    },
    41: {
        "name": "End of no passing",
        "category": "Other",
        "meaning": "The no-overtaking restriction for all vehicles ends here.",
        "driving_tip": "You may now overtake when safe and the road ahead is clear.",
    },
    42: {
        "name": "End no passing veh > 3.5 tons",
        "category": "Other",
        "meaning": "The no-overtaking restriction for heavy vehicles (over 3.5 t) ends here.",
        "driving_tip": "Heavy vehicles may now resume overtaking where the road and traffic conditions permit.",
    },
}

# Sanity check — always 43 entries (class IDs 0–42)
assert len(SIGN_INFO) == 43, f"SIGN_INFO must have 43 entries, found {len(SIGN_INFO)}"
