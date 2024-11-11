import React from 'react';

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

interface PlayerStatsTableProps {
  players: Record<string, PlayerStats>;
}

const PlayerStatsTable: React.FC<PlayerStatsTableProps> = ({ players }) => {
  const sortedPlayers = Object.values(players).sort((a, b) => b.kills - a.kills);

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-200">
          <thead className="text-xs uppercase bg-gray-700">
            <tr>
              <th className="px-4 py-3">Player</th>
              <th className="px-4 py-3">Team</th>
              <th className="px-4 py-3">K</th>
              <th className="px-4 py-3">D</th>
              <th className="px-4 py-3">A</th>
              <th className="px-4 py-3">K/D</th>
              <th className="px-4 py-3">HS%</th>
              <th className="px-4 py-3">DMG</th>
            </tr>
          </thead>
          <tbody>
            {sortedPlayers.map((player, index) => (
              <tr 
                key={player.name}
                className={`
                  ${index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-700'}
                  border-b border-gray-600
                `}
              >
                <td className="px-4 py-3 font-medium">
                  {player.name}
                </td>
                <td className={`px-4 py-3 ${
                  player.team === 'CT' ? 'text-blue-400' : 'text-yellow-400'
                }`}>
                  {player.team}
                </td>
                <td className="px-4 py-3">{player.kills}</td>
                <td className="px-4 py-3">{player.deaths}</td>
                <td className="px-4 py-3">{player.assists}</td>
                <td className="px-4 py-3">{player.kd_ratio}</td>
                <td className="px-4 py-3">{player.hs_percentage}%</td>
                <td className="px-4 py-3">{player.damage}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PlayerStatsTable;