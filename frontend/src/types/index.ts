export interface EngagementData {
  kills: number;
  deaths: number;
  trades: number;
}

export interface AnalysisResult {
  success: boolean;
  data: {
    analysis: {
      engagements: Record<string, EngagementData>;
      positions: Record<string, any>;
    };
  };
}

export interface MapData {
  id: string;
  name: string;
  value: number;
  position: {
    x: number;
    y: number;
    z: number;
  };
  success_rate: number;
  sample_size: number;
}