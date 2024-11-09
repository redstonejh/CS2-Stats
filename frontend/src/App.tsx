import { useState, useEffect } from 'react';
import type { MapData } from './types';
import './index.css';

function App() {
  const [analysisData, setAnalysisData] = useState<MapData[]>([]);

  useEffect(() => {
    // Fetch or generate sample MapData and set it in the state
    const sampleData: MapData[] = [
      {
        id: '1',
        name: 'Location A',
        value: 75,
        position: {
          x: 10,
          y: 20,
          z: 5,
        },
        success_rate: 0.8,
        sample_size: 100,
      },
      {
        id: '2',
        name: 'Location B',
        value: 50,
        position: {
          x: 15,
          y: 30,
          z: 8,
        },
        success_rate: 0.65,
        sample_size: 80,
      },
    ];
    setAnalysisData(sampleData);
  }, []);

  return (
    <div className="min-h-screen bg-cs2-secondary">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-white mb-8">CS2 Stats</h1>
        <div className="bg-white/10 p-4 rounded-lg">
          {analysisData.map((data) => (
            <div key={data.id} className="mb-4">
              <h2 className="text-lg font-medium text-white">{data.name}</h2>
              <p className="text-white">
                Value: {data.value}, Success Rate: {(data.success_rate * 100).toFixed(2)}%, Sample Size: {data.sample_size}
              </p>
              <p className="text-white">
                Position: ({data.position.x}, {data.position.y}, {data.position.z})
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;