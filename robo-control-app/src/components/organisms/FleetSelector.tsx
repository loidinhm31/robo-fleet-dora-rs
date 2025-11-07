import React from "react";
import { Radio, Server } from "lucide-react";
import { FleetStatus } from "../../types";

interface FleetSelectorProps {
  fleetStatus: FleetStatus | null;
  onSelectRover: (entityId: string) => void;
  className?: string;
}

export const FleetSelector: React.FC<FleetSelectorProps> = ({
  fleetStatus,
  onSelectRover,
  className = "",
}) => {
  if (!fleetStatus) {
    return (
      <div className={`p-4 bg-gray-800/50 rounded-lg ${className}`}>
        <div className="flex items-center gap-2 text-gray-400">
          <Server size={16} />
          <span className="text-sm">Loading fleet status...</span>
        </div>
      </div>
    );
  }

  const handleSelect = (entityId: string) => {
    if (entityId !== fleetStatus.selected_entity) {
      onSelectRover(entityId);
    }
  };

  return (
    <div className={`p-4 bg-gray-800/50 rounded-lg ${className}`}>
      <div className="flex items-center gap-2 mb-3">
        <Server size={18} className="text-blue-400" />
        <h3 className="text-sm font-semibold text-gray-200">Fleet Control</h3>
      </div>

      <div className="space-y-2">
        {fleetStatus.fleet_roster.map((entityId) => {
          const isSelected = entityId === fleetStatus.selected_entity;

          return (
            <button
              key={entityId}
              onClick={() => handleSelect(entityId)}
              className={`w-full flex items-center justify-between p-3 rounded-lg transition-all ${
                isSelected
                  ? "bg-blue-600 hover:bg-blue-700 text-white"
                  : "bg-gray-700/50 hover:bg-gray-700 text-gray-300"
              }`}
            >
              <div className="flex items-center gap-2">
                <Radio
                  size={16}
                  className={isSelected ? "text-white" : "text-gray-400"}
                />
                <span className="text-sm font-medium">{entityId}</span>
              </div>

              {isSelected && (
                <span className="text-xs bg-white/20 px-2 py-1 rounded">
                  Active
                </span>
              )}
            </button>
          );
        })}
      </div>

      {fleetStatus.fleet_roster.length === 0 && (
        <div className="text-center py-4">
          <p className="text-sm text-gray-400">No rovers available</p>
        </div>
      )}

      <div className="mt-3 pt-3 border-t border-gray-700">
        <p className="text-xs text-gray-500">
          Selected: <span className="text-gray-300">{fleetStatus.selected_entity}</span>
        </p>
      </div>
    </div>
  );
};
