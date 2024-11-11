export interface Position {
    x: number;
    y: number;
    z: number;
  }
  
  export interface EngagementData {
    position: Position;
    success_rate: number;
    sample_size: number;
  }
  
  export interface MapDimensions {
    width: number;
    height: number;
    gameUnits: number;
  }

  export interface PlayerStats {
    name: string;
    team: string;
    kills: number;
    deaths: number;
    assists: number;
    headshots: number;
    damage: number;
    kd_ratio: number;
    hs_percentage: number;
  }
  
  export interface AnalysisData {
    players: Record<string, PlayerStats>;
    positions: EngagementData[];
    kills: any[];
    rounds: any[];
  }