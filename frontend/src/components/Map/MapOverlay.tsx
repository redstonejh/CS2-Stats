import React, { useState, useEffect } from 'react';
import { Position, EngagementData, MapDimensions } from '../../types/analysis';

interface MapOverlayProps {
  mapName: string;
  analysisType: 'engagements' | 'retakes' | 'executes';
  data: EngagementData[];
  className?: string;
}

// Map dimensions in game units
const MAP_DIMENSIONS: Record<string, MapDimensions> = {
  'de_mirage': {
    width: 1024,
    height: 1024,
    gameUnits: 16384
  },
  'de_inferno': {
    width: 1024,
    height: 1024,
    gameUnits: 16384
  },
  'de_dust2': {
    width: 1024,
    height: 1024,
    gameUnits: 16384
  },
  'de_nuke': {
    width: 1024,
    height: 1024,
    gameUnits: 16384
  },
  'de_ancient': {
    width: 1024,
    height: 1024,
    gameUnits: 16384
  }
};

const HeatmapPoint: React.FC<{
  x: number;
  y: number;
  intensity: number;
  onClick?: () => void;
}> = ({ x, y, intensity, onClick }) => {
  const size = Math.max(20, intensity * 40);
  const opacity = Math.min(0.8, Math.max(0.2, intensity));

  return (
    <div
      className="absolute rounded-full transform -translate-x-1/2 -translate-y-1/2 cursor-pointer
                 transition-all duration-200 hover:scale-110"
      style={{
        left: `${x}%`,
        top: `${y}%`,
        width: `${size}px`,
        height: `${size}px`,
        background: `radial-gradient(circle, rgba(59,130,246,${opacity}) 0%, rgba(59,130,246,0) 70%)`,
      }}
      onClick={onClick}
    />
  );
};

export const MapOverlay: React.FC<MapOverlayProps> = ({
  mapName,
  analysisType,
  data,
  className = ''
}) => {
  const [selectedPoint, setSelectedPoint] = useState<EngagementData | null>(null);
  const [dimensions, setDimensions] = useState<MapDimensions | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setDimensions(MAP_DIMENSIONS[mapName]);
    setIsLoading(false);
  }, [mapName]);

  const convertGameUnitsToPercentage = (position: Position): { x: number; y: number } => {
    if (!dimensions) return { x: 0, y: 0 };

    return {
      x: (position.x / dimensions.gameUnits) * 100,
      y: (position.y / dimensions.gameUnits) * 100
    };
  };

  const renderDataPoints = () => {
    return data.map((point, index) => {
      const { x, y } = convertGameUnitsToPercentage(point.position);
      const intensity = point.success_rate;

      return (
        <HeatmapPoint
          key={index}
          x={x}
          y={y}
          intensity={intensity}
          onClick={() => setSelectedPoint(point)}
        />
      );
    });
  };

  const getAnalysisTypeLabel = () => {
    switch (analysisType) {
      case 'engagements':
        return 'Engagement Heatmap';
      case 'retakes':
        return 'Retake Success Rates';
      case 'executes':
        return 'Execute Analysis';
      default:
        return 'Analysis';
    }
  };

  if (isLoading || !dimensions) {
    return (
      <div className="w-full h-64 bg-gray-800 rounded-lg flex items-center justify-center">
        <div className="text-gray-400">Loading map...</div>
      </div>
    );
  }

  return (
    <div className={`relative bg-gray-800 rounded-lg overflow-hidden ${className}`}>
      {/* Map Header */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-gray-900/75 p-4 flex justify-between items-center">
        <div className="text-white font-semibold">{mapName}</div>
        <div className="text-blue-400">{getAnalysisTypeLabel()}</div>
      </div>

      {/* Map Container */}
      <div className="relative w-full" style={{ paddingTop: '100%' }}>
        {/* Map Image */}
        <div 
          className="absolute inset-0 bg-center bg-cover"
          style={{
            backgroundImage: `url(/maps/${mapName}_radar.png)`
          }}
        />

        {/* Data Overlay */}
        <div className="absolute inset-0">
          {renderDataPoints()}
        </div>
      </div>

      {/* Analysis Panel */}
      {selectedPoint && (
        <div className="absolute bottom-0 left-0 right-0 bg-gray-900/75 p-4 text-white">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-lg font-semibold mb-2">Position Analysis</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-gray-400">Success Rate</div>
                  <div className="text-lg">{(selectedPoint.success_rate * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-sm text-gray-400">Sample Size</div>
                  <div className="text-lg">{selectedPoint.sample_size}</div>
                </div>
              </div>
            </div>
            <button
              className="text-gray-400 hover:text-white"
              onClick={() => setSelectedPoint(null)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute top-4 right-4 bg-gray-900/75 p-2 rounded text-white text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500/80" />
          <span>High Activity</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500/20" />
          <span>Low Activity</span>
        </div>
      </div>
    </div>
  );
};