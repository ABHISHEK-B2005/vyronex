/**
 * Vyronex Activity Registry (Single Source of Truth)
 */

export const VYRONEX_ACTIVITIES = {
  // Gym / Home Workout Activities
  squat: {
    id: "squat",
    label: "Bodyweight Squat",
    category: "gym",
    engine: "squat_analysis",
    icon: "🏋️",
    metrics: ["knee_alignment", "knee_depth", "back_posture", "symmetry"],
    description: "Screen knee valgus (cave-in), lumbar flexion, and depth symmetry.",
    instructions: [
      "Face the camera from 6-8 feet away, with your full body visible.",
      "Keep feet shoulder-width apart and toes angled slightly outward.",
      "Lower hips until thighs are parallel to the floor, driving knees outward over toes.",
    ],
  },
  pushup: {
    id: "pushup",
    label: "Standard Push-Up",
    category: "gym",
    engine: "pushup_analysis",
    icon: "💪",
    metrics: ["elbow_flare", "shoulder_alignment", "body_line"],
    description: "Check scapular winging, elbow flare angle, and core sagging.",
    instructions: [
      "Position camera at a 45-degree angle from the side.",
      "Maintain a straight line from ears to ankles throughout the rep.",
      "Tuck elbows to 45 degrees rather than flaring straight out.",
    ],
  },
  lunge: {
    id: "lunge",
    label: "Walking Lunge",
    category: "gym",
    engine: "lunge_analysis",
    icon: "🚶",
    metrics: ["knee_alignment", "torso_posture", "hip_symmetry"],
    description: "Analyze lead knee alignment, torso uprightness, and hip levelness.",
    instructions: [
      "Step forward smoothly, lowering back knee toward the floor.",
      "Keep front knee stacked directly above the front ankle.",
      "Keep torso tall without leaning forward onto the thigh.",
    ],
  },
  plank: {
    id: "plank",
    label: "Forearm Plank",
    category: "gym",
    engine: "plank_analysis",
    icon: "⏱️",
    metrics: ["hip_alignment", "back_posture", "hold_duration"],
    description: "Evaluate pelvic tilt, scapular depression, and spinal neutral.",
    instructions: [
      "Place forearms on floor with elbows directly under shoulders.",
      "Brace core 360 degrees, keeping hips level with shoulders.",
      "Hold steady without arching the lower back.",
    ],
  },

  // Track & Field Athletics Activities
  long_jump: {
    id: "long_jump",
    label: "Long Jump",
    category: "athlete",
    engine: "long_jump_analysis",
    icon: "🏃‍♂️",
    metrics: ["takeoff_angle", "landing_symmetry", "impact_control"],
    description: "Assess takeoff knee angle, torso inclination, and landing asymmetry.",
    instructions: [
      "Position camera at the takeoff board / sand pit boundary.",
      "Drive the lead knee up explosively at takeoff (165° target angle).",
      "Land symmetrically with both knees absorbing impact equally.",
    ],
  },
  shot_put: {
    id: "shot_put",
    label: "Shot Put",
    category: "athlete",
    engine: "shot_put_analysis",
    icon: "☄️",
    metrics: ["trunk_control", "knee_extension", "release_arc"],
    description: "Track glide trunk inclination, delivery knee extension, and release arc.",
    instructions: [
      "Position camera capturing stance to release along the throwing circle.",
      "Maintain low trunk inclination during the glide across the ring.",
      "Extend rear leg explosively into the final delivery push.",
    ],
  },
  javelin: {
    id: "javelin",
    label: "Javelin Throw",
    category: "athlete",
    engine: "javelin_analysis",
    icon: "🎯",
    metrics: ["brace_leg", "elbow_alignment", "trunk_rotation"],
    description: "Monitor front brace leg stiffness and throwing elbow angle.",
    instructions: [
      "Capture the final crossover steps and plant phase from 45 degrees.",
      "Plant front brace leg firmly (> 160° angle) without knee collapse.",
      "Keep elbow high above shoulder height throughout the pull.",
    ],
  },
};

export function getActivity(id) {
  return VYRONEX_ACTIVITIES[id] || null;
}

export function getActivitiesByCategory(category) {
  return Object.values(VYRONEX_ACTIVITIES).filter(a => a.category === category);
}
