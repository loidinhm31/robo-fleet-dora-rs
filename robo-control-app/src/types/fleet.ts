// Fleet management types

export interface FleetStatus {
  selected_entity: string;
  fleet_roster: string[];
  timestamp: number;
}

export interface FleetSelectCommand {
  entity_id: string;
  timestamp: number;
}

export interface RoverStatus {
  entity_id: string;
  is_selected: boolean;
  is_connected: boolean;
  last_seen?: number;
  battery_level?: number;
  signal_strength?: number;
}

export interface FleetRosterUpdate {
  rovers: RoverStatus[];
  timestamp: number;
}

// Helper function to create fleet select command
export function createFleetSelectCommand(entityId: string): FleetSelectCommand {
  return {
    entity_id: entityId,
    timestamp: Date.now(),
  };
}
