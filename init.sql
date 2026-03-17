CREATE TABLE IF NOT EXISTS exercice (
    id SERIAL PRIMARY KEY,
    name VARCHAR UNIQUE,
    description TEXT,
    duration_time FLOAT
);

INSERT INTO exercice (name, description, duration_time) VALUES
(
  'Shoulder Abduction',
  'The elbow is a hinge joint in the upper limb formed by the articulation of three bones: the humerus (upper arm bone), radius (forearm bone on the thumb side), and ulna (forearm bone on the little finger side). It allows flexion and extension of the arm, enabling movements like bending and straightening.',
  5
),
(
  'arm abduction',
  'Arm abduction is the movement of the arm away from the body midline in the coronal (frontal) plane. It begins with the arm at the side and progresses upward, such as when raising the arm to the side or overhead. This motion is initiated by the supraspinatus (first 15°), continued by the deltoid (up to 90°), and completed by upward rotation of the scapula driven by the trapezius and serratus anterior beyond 90°. Full abduction can reach 160–180° and is essential for activities like performing a jumping jack.',
  5
)
ON CONFLICT (name) DO NOTHING;
