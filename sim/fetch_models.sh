#!/usr/bin/env bash
# Pre-download the Gazebo Fuel models used by tracker_world.sdf so the world loads quickly.
# Needs internet once; models are cached in ~/.gz/fuel afterwards.
set -u
BASE=https://fuel.gazebosim.org/1.0/OpenRobotics/models
for m in "Walking%20person" "Casual%20female" "Standing%20person" Pickup SUV Hatchback Bus "Oak%20tree" "Construction%20Cone"; do
  echo "== $m"
  gz fuel download -u "$BASE/$m" || echo "WARN: failed to download $m"
done
