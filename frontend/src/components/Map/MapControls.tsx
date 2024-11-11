import React from 'react';

interface MapControlsProps {
  mapName: string;
  onMapChange: (map: string) => void;
  analysisType: string;
  onAnalysisTypeChange: (type: string) => void;
}

export const MapControls: React.FC<MapControlsProps> = ({
  mapName,
  onMapChange,
  analysisType,
  onAnalysisTypeChange
}) => {
  const maps = [
    'de_mirage',
    'de_inferno',
    'de_dust2',
    'de_nuke',
    'de_ancient'
  ];

  const analysisTypes = [
    { id: 'engagements', label: 'Engagements' },
    { id: 'retakes', label: 'Retakes' },
    { id: 'executes', label: 'Executes' }
  ];

  return (
    <div className="flex flex-col gap-4 mb-4">
      {/* Map Selection */}
      <div className="flex flex-wrap gap-2">
        {maps.map(map => (
          <button
            key={map}
            onClick={() => onMapChange(map)}
            className={`
              px-4 py-2 rounded-lg text-sm font-medium transition-colors
              ${mapName === map 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-700 text-gray-200 hover:bg-gray-600'}
            `}
          >
            {map.replace('de_', '')}
          </button>
        ))}
      </div>

      {/* Analysis Type Selection */}
      <div className="flex gap-2">
        {analysisTypes.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => onAnalysisTypeChange(id)}
            className={`
              px-4 py-2 rounded-lg text-sm font-medium transition-colors
              ${analysisType === id 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-700 text-gray-200 hover:bg-gray-600'}
            `}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
};