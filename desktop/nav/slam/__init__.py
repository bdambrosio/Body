"""SLAM: IMU-assisted scan matching for the Body robot.

Pose pipeline (see docs/encoder_integration_spec.md +
docs/imu_integration_spec.md):
- Rotation: BNO085 on-chip fusion → yaw quaternion.
- Translation: 2D lidar scan-to-map correlation matching.
- Encoders (secondary): stall / slip detection, translation sanity.

Modules here are designed to be exercised with synthetic data so the
algorithms can be developed and validated while hardware is offline.
When live data is available, ImuPlusScanMatchPose drops into the
world_map fuser's PoseSource slot with no fusion-path rewrites.
"""
