import React from "react";
import { SliderControl } from "../molecules";
import { JointPositions, JOINT_LIMITS } from "../../types";

export interface JointControlPanelProps {
  jointPositions: JointPositions;
  onJointChange: (joint: keyof JointPositions, value: number) => void;
  disabled?: boolean;
  className?: string;
}

const JOINT_LABELS: Record<keyof typeof JOINT_LIMITS, string> = {
  shoulder_pan: "Shoulder Pan",
  shoulder_lift: "Shoulder Lift",
  elbow_flex: "Elbow Flex",
  wrist_flex: "Wrist Flex",
  wrist_roll: "Wrist Roll",
  gripper: "Gripper",
};

export const JointControlPanel: React.FC<JointControlPanelProps> = ({
  jointPositions,
  onJointChange,
  disabled = false,
  className = "",
}) => {
  return (
    <div className={`grid grid-cols-1 md:grid-cols-2 gap-3 md:gap-4 ${className}`}>
      {(Object.keys(JOINT_LIMITS) as Array<keyof typeof JOINT_LIMITS>).map((joint) => (
        <SliderControl
          key={joint}
          label={JOINT_LABELS[joint]}
          value={jointPositions[joint] || 0}
          min={JOINT_LIMITS[joint].min}
          max={JOINT_LIMITS[joint].max}
          unit="rad"
          onChange={(value) => onJointChange(joint, value)}
          disabled={disabled}
          decimals={2}
        />
      ))}
    </div>
  );
};
