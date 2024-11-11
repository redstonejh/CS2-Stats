import { useState } from 'react';
import { MapOverlay } from './components/Map/MapOverlay';
import { MapControls } from './components/Map/MapControls';
import PlayerStatsTable from './components/Analysis/PlayerStatsTable';
import type { EngagementData } from './types/analysis';

// Define the analysis data type
interface AnalysisData {
  players: Record<string, PlayerStats>;
  positions: EngagementData[];
  kills: any[];
  rounds: any[];
}

interface PlayerStats {
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

function App() {
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [selectedMap, setSelectedMap] = useState('de_mirage');
  const [analysisType, setAnalysisType] = useState<'engagements' | 'retakes' | 'executes'>('engagements');

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-gray-900 dark:text-white">
          CS2 Analytics
        </h1>
        
        {/* Map Controls */}
        <MapControls
          mapName={selectedMap}
          onMapChange={setSelectedMap}
          analysisType={analysisType}
          onAnalysisTypeChange={(type) => setAnalysisType(type as 'engagements' | 'retakes' | 'executes')}
        />
        
        {analysis && (
          <div className="mt-8 space-y-8">
            <div>
              <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                Player Statistics
              </h2>
              <PlayerStatsTable players={analysis.players} />
            </div>
            
            <div>
              <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                Match Analysis
              </h2>
              <MapOverlay 
                mapName={selectedMap}
                analysisType={analysisType}
                data={analysis.positions}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;