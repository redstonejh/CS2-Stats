from __future__ import annotations
from dataclasses import dataclass, field
from typing import Counter, Dict, List, Optional, Tuple, Set, Any, BinaryIO, Literal, Union
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import struct
import logging
import json
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Team(Enum):
    """Team enumeration for CS2"""
    CT = "CT"
    T = "T"
    SPECTATOR = "SPECTATOR"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, value: str) -> Team:
        """Safely convert string to Team enum"""
        try:
            return cls(value.upper())
        except (ValueError, AttributeError):
            return cls.UNKNOWN

class EventType(Enum):
    """Game event types"""
    ROUND_START = auto()
    ROUND_END = auto()
    BOMB_PLANTED = auto()
    BOMB_DEFUSED = auto()
    PLAYER_DEATH = auto()
    PLAYER_POSITION = auto()
    SMOKE = auto()
    FLASH = auto()
    MOLOTOV = auto()
    GAME_PHASE_CHANGE = auto()

    @property
    def is_utility(self) -> bool:
        """Check if event is utility-based"""
        return self in {self.SMOKE, self.FLASH, self.MOLOTOV}

@dataclass
class TickData:
    """Represents a game server tick with timing information"""
    number: int
    time: float
    interval: float = 0.015625  # Default CS2 tick rate (64 tick)

    @property
    def milliseconds(self) -> float:
        """Convert tick to milliseconds"""
        return self.time * 1000

    @classmethod
    def from_time(cls, time: float, tick_rate: int = 64) -> 'TickData':
        """Create TickData from a time value"""
        interval = 1 / tick_rate
        tick_number = int(time / interval)
        return cls(number=tick_number, time=time, interval=interval)

@dataclass(frozen=True)
class Position:
    """Immutable 3D position"""
    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional[Position]:
        """Create position from dictionary"""
        try:
            return cls(
                x=float(data.get('x', 0)),
                y=float(data.get('y', 0)),
                z=float(data.get('z', 0))
            )
        except (TypeError, ValueError):
            logger.warning(f"Invalid position data: {data}")
            return None

    @classmethod
    def from_tuple(cls, pos: Tuple[float, float, float]) -> Position:
        """Create position from tuple"""
        return cls(x=pos[0], y=pos[1], z=pos[2])

    def distance_to(self, other: Position) -> float:
        """Calculate Euclidean distance to another position"""
        return ((self.x - other.x) ** 2 + 
                (self.y - other.y) ** 2 + 
                (self.z - other.z) ** 2) ** 0.5

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation"""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }

@dataclass
class PlayerState:
    """Track player state at a specific tick"""
    player_id: int
    name: str
    team: Team
    position: Position
    health: int = 100
    armor: int = 100
    is_alive: bool = True
    has_helmet: bool = False
    has_defuser: bool = False
    active_weapon: str = ""
    tick: int = 0

@dataclass
class GameEvent:
    """Game event with complete state information"""
    tick: int
    event_type: EventType
    data: Dict[str, Any]
    position: Optional[Position]
    round_number: int
    timestamp: float
    highlighted: bool = False

    @property
    def is_kill_event(self) -> bool:
        """Check if event is a kill"""
        return self.event_type == EventType.PLAYER_DEATH

    @property
    def is_utility_event(self) -> bool:
        """Check if event is utility usage"""
        return self.event_type.is_utility

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'tick': self.tick,
            'type': self.event_type.name,
            'data': self.data,
            'position': self.position.to_dict() if self.position else None,
            'round': self.round_number,
            'timestamp': self.timestamp,
            'highlighted': self.highlighted
        }

@dataclass
class Round:
    """Complete round information"""
    number: int
    start_tick: int
    end_tick: int
    winner: Team
    events: List[GameEvent] = field(default_factory=list)
    player_states: Dict[int, List[PlayerState]] = field(default_factory=lambda: defaultdict(list))
    score_ct: int = 0
    score_t: int = 0

    @property
    def duration_ticks(self) -> int:
        """Get round duration in ticks"""
        return self.end_tick - self.start_tick

    @property
    def bomb_planted(self) -> bool:
        """Check if bomb was planted this round"""
        return any(e.event_type == EventType.BOMB_PLANTED for e in self.events)

    @property
    def plant_tick(self) -> Optional[int]:
        """Get tick when bomb was planted"""
        plant_event = next((e for e in self.events if e.event_type == EventType.BOMB_PLANTED), None)
        return plant_event.tick if plant_event else None

class DemoHeader:
    """CS2 Demo file header information"""
    def __init__(self, raw_data: bytes):
        self.magic: str = raw_data[:8].decode('ascii').strip('\0')
        self.demo_protocol: int = struct.unpack('i', raw_data[8:12])[0]
        self.network_protocol: int = struct.unpack('i', raw_data[12:16])[0]
        self.server_name: str = raw_data[16:272].decode('ascii').strip('\0')
        self.client_name: str = raw_data[272:528].decode('ascii').strip('\0')
        self.map_name: str = raw_data[528:784].decode('ascii').strip('\0')
        self.game_directory: str = raw_data[784:1040].decode('ascii').strip('\0')
        self.playback_time: float = struct.unpack('f', raw_data[1040:1044])[0]
        self.ticks: int = struct.unpack('i', raw_data[1044:1048])[0]
        self.frames: int = struct.unpack('i', raw_data[1048:1052])[0]
        self.signon_length: int = struct.unpack('i', raw_data[1052:1056])[0]

class DemoParser:
    """CS2 Demo Parser with comprehensive analysis capabilities"""
    
    HEADER_SIZE = 1056
    TRADE_WINDOW_TICKS = 150
    PLAYERS_PER_TEAM = 5
    
    def __init__(self, demo_path: str):
        """Initialize parser with demo file"""
        self.demo_path = Path(demo_path)
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        
        self.header: Optional[DemoHeader] = None
        self.rounds: List[Round] = []
        self.players: Dict[int, PlayerState] = {}
        self.current_round: int = 0
        self.events: List[GameEvent] = []
        self.string_tables: Dict[int, Dict[int, str]] = {}
        self.data_tables: Dict[int, Dict[int, Any]] = {}
        
        # Analysis caches
        self._retake_cache: Dict[int, bool] = {}
        self._position_cache: Dict[Tuple[float, float, float], bool] = {}
        
        logger.info(f"Initialized parser for {demo_path}")
    
    def _read_header(self):
        with self.demo_path.open('rb') as demo_file:
            header_data = demo_file.read(self.HEADER_SIZE)
            self.header = DemoHeader(header_data)
    
    def _parse_demo(self):
        with self.demo_path.open('rb') as demo_file:
            demo_file.seek(self.HEADER_SIZE)
            while True:
                packet = self._read_next_packet(demo_file)
                if not packet:
                    break
                self._process_packet(packet)

    def _process_packet(self, packet: bytes):
        events = self._decode_packet(packet)
        for event in events:
            self._handle_event(event)
    
    def _handle_event(self, event: Dict[str, Any]):
        """Handle a single game event"""
        event_type = EventType[event['type']]
        if event_type == EventType.ROUND_START:
            self._start_new_round(event)
        elif event_type == EventType.ROUND_END:
            self._end_round(event)
        else:
            self._add_event(event)
    
    def _start_new_round(self, event: Dict[str, Any]):
        """Initialize a new round"""
        self.current_round += 1
        self.rounds.append(Round(
            number=self.current_round,
            start_tick=event['tick'],
            end_tick=0,
            winner=Team.from_string(event['data']['winner'])
        ))
    
    def _end_round(self, event: Dict[str, Any]):
        """Finish the current round"""
        current_round = self.rounds[-1]
        current_round.end_tick = event['tick']

    def _add_event(self, event: Dict[str, Any]):
        """Add a game event to the current round"""
        current_round = self.rounds[-1]
        position = None
        if 'x' in event and 'y' in event and 'z' in event:
            position = Position(x=event['x'], y=event['y'], z=event['z'])
        game_event = GameEvent(
            tick=event['tick'],
            event_type=EventType[event['type']],
            data=event['data'],
            position=position,
            round_number=current_round.number,
            timestamp=TickData.from_time(event['tick'] * 0.015625).milliseconds
        )
        current_round.events.append(game_event)
        self._update_player_state(game_event)
    
    def _update_player_state(self, event: GameEvent):
        """Update player state based on a game event"""
        if event.event_type == EventType.PLAYER_POSITION:
            self._update_player_position(event)
        elif event.event_type == EventType.PLAYER_DEATH:
            self._handle_player_death(event)
        elif event.event_type == EventType.BOMB_PLANTED:
            self._handle_bomb_planted(event)
        elif event.event_type == EventType.BOMB_DEFUSED:
            self._handle_bomb_defused(event)
    
    def _update_player_position(self, event: GameEvent):
        """Update player position based on a PLAYER_POSITION event"""
        player_id = event.data['player_id']
        if player_id not in self.players:
            self.players[player_id] = PlayerState(
                player_id=player_id,
                name=event.data['player_name'],
                team=Team.from_string(event.data['team']),
                position=event.position or Position(0, 0, 0),  # Use default position if event.position is None
                tick=event.tick
            )
        else:
            self.players[player_id].position = event.position or self.players[player_id].position  # Update position if available, otherwise keep the existing position
            self.players[player_id].tick = event.tick

    def _handle_player_death(self, event: GameEvent):
        """Handle a PLAYER_DEATH event"""
        killer_id = event.data['killer_id']
        victim_id = event.data['victim_id']

        if killer_id in self.players:
            self.players[killer_id].health = 100
            self.players[killer_id].armor = 100

        if victim_id in self.players:
            self.players[victim_id].is_alive = False
    
    def _handle_bomb_planted(self, event: GameEvent):
        """Handle a BOMB_PLANTED event"""
        # Update player state, round state, etc.
        pass

    def _handle_bomb_defused(self, event: GameEvent):
        """Handle a BOMB_DEFUSED event"""
        # Update player state, round state, etc.
        pass
    
    
    def _analyze(self) -> Dict[str, Any]:
        """Perform analysis on the parsed demo"""
        try:
            analysis = {
                'total_rounds': len(self.rounds),
                'round_winners': self._get_round_winners(),
                'total_kills': self._get_total_kills(),
                'kill_counts': self._get_kill_counts(),
                'player_stats': self._get_player_stats()
            }
            return analysis
        except Exception as e:
            logger.error(f"Error performing analysis: {e}")
            return {}
        
    def _get_round_winners(self) -> Dict[str, int]:
        """Get the number of rounds won by each team"""
        winners = Counter(round.winner.value for round in self.rounds)
        return dict(winners)
    
    def _get_total_kills(self) -> int:
        """Get the total number of kills in the demo"""
        return sum(event.data['victim_id'] is not None for event in self.events if event.event_type == EventType.PLAYER_DEATH)
    
    def _get_kill_counts(self) -> Dict[int, int]:
        """Get the kill counts for each player"""
        kill_counts = Counter(event.data['killer_id'] for event in self.events if event.event_type == EventType.PLAYER_DEATH)
        return dict(kill_counts)
    
    def _get_player_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed stats for each player"""
        player_stats = {}
        for player_id, player_states in self._get_player_states().items():
            average_health, average_armor = self._calculate_player_averages(player_states)
            stats = {
                'total_kills': self._get_player_kills(player_id),
                'total_deaths': self._get_player_deaths(player_id),
                'total_damage': self._get_player_damage(player_id),
                'headshot_kills': self._get_player_headshot_kills(player_id),
                'average_health': average_health,
                'average_armor': average_armor
            }
            player_stats[player_id] = stats
        return player_stats
    
    def _get_player_states(self) -> Dict[int, List[PlayerState]]:
        """Get a dictionary of player IDs to lists of player states"""
        player_states = defaultdict(list)
        for round in self.rounds:
            for event in round.events:
                if event.event_type == EventType.PLAYER_POSITION:
                    player_id = event.data['player_id']
                    player_states[player_id].append(PlayerState(
                        player_id=player_id,
                        name=event.data['player_name'],
                        team=Team.from_string(event.data['team']),
                        position=event.position or Position(0, 0, 0),
                        tick=event.tick
                    ))
        return player_states
    
    def _calculate_player_averages(self, player_states: List[PlayerState]) -> Tuple[float, float]:
        """Calculate the average health and armor for a list of player states"""
        total_health = sum(state.health for state in player_states)
        total_armor = sum(state.armor for state in player_states)
        num_states = len(player_states)
        return (total_health / num_states if num_states > 0 else 0.0,
                total_armor / num_states if num_states > 0 else 0.0)
    
    def _get_player_headshot_kills(self, player_id: int) -> int:
        """Get the number of headshot kills for a player"""
        headshot_kills = 0
        for event in self.events:
            if event.event_type == EventType.PLAYER_DEATH and event.data['killer_id'] == player_id:
                if event.data['is_headshot']:
                    headshot_kills += 1
        return headshot_kills

    def _get_player_damage(self, player_id: int) -> int:
        """Get the total damage dealt by a player"""
        total_damage = 0
        for event in self.events:
            if event.event_type == EventType.PLAYER_DEATH:
                if event.data['killer_id'] == player_id:
                    total_damage += event.data['victim_health']
                if event.data['victim_id'] == player_id:
                    total_damage += event.data['victim_health']
        return total_damage
    
    def _get_player_kills(self, player_id: int) -> int:
        """Get the total kills for a player"""
        return sum(event.data['killer_id'] == player_id for event in self.events if event.event_type == EventType.PLAYER_DEATH)
    
    def _get_player_deaths(self, player_id: int) -> int:
        """Get the total deaths for a player"""
        return sum(event.data['victim_id'] == player_id for event in self.events if event.event_type == EventType.PLAYER_DEATH)
    
    def _save_analysis(self, analysis: Dict[str, Any]):
        """Save the analysis results"""
        try:
            # Construct the output file path
            output_dir = self.demo_path.parent / 'analysis'
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{self.demo_path.stem}.json"

            # Save the analysis data to a JSON file
            with output_file.open('w') as f:
                json.dump(analysis, f, indent=4)

            logger.info(f"Analysis saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")


    def parse(self) -> Dict[str, Any]:
        """Parse demo file and return analysis"""
        try:
            self._read_header()
            self._parse_demo()
            analysis = self._analyze()
            self._save_analysis(analysis)
            return analysis
        except Exception as e:
            logger.error(f"Error parsing demo: {e}", exc_info=True)
            raise

    def _read_next_packet(self, demo_file: BinaryIO) -> Optional[bytes]:
        try:
            cmd_info = self._read_cmd_info(demo_file)
            if not cmd_info:
                return None
            cmd_type, tick, size = cmd_info

            packet_data = demo_file.read(size)
            if len(packet_data) < size:
                logger.warning(f"Incomplete packet data: expected {size} bytes, got {len(packet_data)}")
                return None
            packet = self._create_packet(cmd_type, tick, packet_data)
            return packet
        
        except EOFError:
            return None
        except Exception as e:
            logger.error(f"Error reading packet: {e}")
            return None
    
    def _read_cmd_info(self, demo_file: BinaryIO) -> Optional[Tuple[int, int, int]]:
        try:
            cmd_type_data = demo_file.read(1)
            if not cmd_type_data:
                return None
            cmd_type = int.from_bytes(cmd_type_data, byteorder='little')

            tick_data = demo_file.read(4)
            if not tick_data:
                return None
            tick = int.from_bytes(tick_data, byteorder='little')

            size_data = demo_file.read(4)
            if not size_data:
                return None
            size = int.from_bytes(size_data, byteorder='little')

            return (cmd_type, tick, size)
        except EOFError:
            return None
        
    def _create_packet(self, cmd_type: int, tick: int, data: bytes) -> bytes:
        header = struct.pack('<BII', cmd_type, tick, len(data))
        return header + data
    
    def _decode_packet(self, packet: bytes) -> List[Dict[str, Any]]:
        events = []
        try:
            cmd_type, tick, size = struct.unpack('<BII', packet[:9])
            data = packet[9:]

            if cmd_type == self.CMD_PACKET:
                events.extend(self._parse_game_events(data, tick))
            elif cmd_type == self.CMD_STRINGTABLES:
                self._update_string_tables(data)
            elif cmd_type == self.CMD_DATATABLES:
                self._update_data_tables(data)

        except Exception as e:
            logger.error(f"Error decoding packet: {e}")

        return events
    
    CMD_PACKET = 1
    CMD_STRINGTABLES = 2
    CMD_DATATABLES = 3

    def _parse_game_events(self, data: bytes, tick: int) -> List[Dict[str, Any]]:
        events = []
        offset = 0
        try:
            while offset < len(data):
                # Read event header
                event_type, event_size = struct.unpack('<BB', data[offset:offset+2])
                offset += 2
            
                # Read event data
                event_data = data[offset:offset+event_size]
                offset += event_size
            
                # Parse event
                event = self._parse_event(event_type, event_data, tick)
                if event:
                    events.append(event)

        except Exception as e:
            logger.error(f"Error parsing game events: {e}")  

        return events
    
    EVENT_TYPE_MAP = {
    0: EventType.ROUND_START,
    1: EventType.ROUND_END,
    2: EventType.BOMB_PLANTED,
    3: EventType.BOMB_DEFUSED,
    4: EventType.PLAYER_DEATH,
    5: EventType.PLAYER_POSITION,
    6: EventType.SMOKE,
    7: EventType.FLASH,
    8: EventType.MOLOTOV,
    9: EventType.GAME_PHASE_CHANGE
}
    def _parse_event(self, event_type: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], event_data: bytes, tick: int) -> Optional[Dict[str, Any]]:
        try:
            # Map event type to EventType enum
            if event_type not in self.EVENT_TYPE_MAP:
                return None
            event_name = self.EVENT_TYPE_MAP[event_type].name

            parsed_data = self._parse_event_data(event_type, event_data)
            if not parsed_data:
                return None
        
            event = {
                'type': event_name,
                'tick': tick,
                'data': parsed_data
            }

            if 'x' in parsed_data and 'y' in parsed_data and 'z' in parsed_data:
                event['position'] = {
                    'x': parsed_data['x'],
                    'y': parsed_data['y'],
                    'z': parsed_data['z']
                }
            return event
    
        except Exception as e:
            logger.error(f"Error parsing event: {e}")
            return None
    
    def _parse_event_data(self, event_type: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], event_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse event data based on event type"""
        try:
            if event_type == 0:
                return self._parse_round_start_event(event_data)
            elif event_type == 1:
                return self._parse_round_end_event(event_data)
            elif event_type == 2:
                return self._parse_bomb_planted_event(event_data)
            elif event_type == 3:
                return self._parse_bomb_defused_event(event_data)
            elif event_type == 4:
                return self._parse_player_death_event(event_data)
            elif event_type == 5:
                return self._parse_player_position_event(event_data)
            elif event_type == 6:
                return self._parse_smoke_event(event_data)
            elif event_type == 7:
                return self._parse_flash_event(event_data)
            elif event_type == 8:
                return self._parse_molotov_event(event_data)
            elif event_type == 9:
                return self._parse_game_phase_change_event(event_data)
        except Exception as e:
            logger.error(f"Error parsing event data: {e}")
            return None
    
    def _parse_molotov_event(self, event_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse MOLOTOV event data"""
        try:
            player_id = self._read_int(event_data, 0)
            position = self._read_position(event_data, 4)
            duration = self._read_float(event_data, 16)
            return {
                'player_id': player_id,
                'position': position.to_dict(),
                'duration': duration
            }
        except Exception as e:
            logger.error(f"Error parsing MOLOTOV event data: {e}")
            return None
    
    def _parse_game_phase_change_event(self, event_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse GAME_PHASE_CHANGE event data"""
        try:
            phase = self._read_string(event_data, 0)
            return {
                'phase': phase
            }
        except Exception as e:
            logger.error(f"Error parsing GAME_PHASE_CHANGE event data: {e}")
            return None
        
    def _parse_flash_event(self, event_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse FLASH event data"""
        try:
            player_id = self._read_int(event_data, 0)
            position = self._read_position(event_data, 4)
            duration = self._read_float(event_data, 16)
            return {
                'player_id': player_id,
                'position': position.to_dict(),
                'duration': duration
            }
        except Exception as e:
            logger.error(f"Error parsing FLASH event data: {e}")
            return None

    def _parse_round_start_event(self, event_data: bytes) -> Dict[str, Any]:
        """Parse ROUND_START event data"""
        # Implement round start event parsing logic
        return {
            'winner': self._read_string(event_data)
        }
    
    def _parse_round_end_event(self, event_data: bytes) -> Dict[str, Any]:
        """Parse ROUND_END event data"""
        # Implement round end event parsing logic
        return {
            'winner': self._read_string(event_data)
        }
    def _parse_bomb_defused_event(self, event_data: bytes) -> Dict[str, Any]:
        """Parse BOMB_DEFUSED event data"""
        player_id = self._read_int(event_data, 0)
        position = self._read_position(event_data, 4)
        return {
            'player_id': player_id,
            'position': position.to_dict()
        }
    
    def _parse_bomb_planted_event(self, event_data: bytes) -> Dict[str, Any]:
        """Parse BOMB_PLANTED event data"""
        # Implement bomb planted event parsing logic
        return {
            'player_id': self._read_int(event_data, 4),
            'position': self._read_position(event_data, 8)
        }
    
    def _parse_player_death_event(self, event_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse PLAYER_DEATH event data"""
        try:
            victim_id = self._read_int(event_data, 0)
            killer_id = self._read_int(event_data, 4)
            assister_id = self._read_int(event_data, 8)
            victim_health = self._read_int(event_data, 12)
            killer_health = self._read_int(event_data, 16)
            assister_health = self._read_int(event_data, 20)
            victim_armor = self._read_int(event_data, 24)
            killer_armor = self._read_int(event_data, 28)
            assister_armor = self._read_int(event_data, 32)
            victim_position = self._read_position(event_data, 36)
            killer_position = self._read_position(event_data, 48)
            assister_position = self._read_position(event_data, 60)
            is_headshot = self._read_bool(event_data, 72)
            return {
                'victim_id': victim_id,
                'killer_id': killer_id,
                'assister_id': assister_id,
                'victim_health': victim_health,
                'killer_health': killer_health,
                'assister_health': assister_health,
                'victim_armor': victim_armor,
                'killer_armor': killer_armor,
                'assister_armor': assister_armor,
                'victim_position': victim_position.to_dict(),
                'killer_position': killer_position.to_dict(),
                'assister_position': assister_position.to_dict(),
                'is_headshot': is_headshot
            }
        except Exception as e:
            logger.error(f"Error parsing PLAYER_DEATH event data: {e}")
            return None
    
    def _parse_player_position_event(self, event_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse PLAYER_POSITION event data"""
        try:
            player_id = self._read_int(event_data, 0)
            player_name = self._read_string(event_data, 4)
            team = self._read_string(event_data, 36)
            position = self._read_position(event_data, 68)
            return {
                'player_id': player_id,
                'player_name': player_name,
                'team': team,
                'position': position.to_dict()
            }
        except Exception as e:
            logger.error(f"Error parsing PLAYER_POSITION event data: {e}")
            return None
    def _parse_smoke_event(self, event_data: bytes) -> Optional[Dict[str, Any]]:
        """Parse SMOKE event data"""
        try:
            player_id = self._read_int(event_data, 0)
            position = self._read_position(event_data, 4)
            duration = self._read_float(event_data, 16)
            return {
                'player_id': player_id,
                'position': position.to_dict(),
                'duration': duration
            }
        except Exception as e:
            logger.error(f"Error parsing SMOKE event data: {e}")
            return None

    def _read_string(self, data: bytes, offset: int = 0) -> str:
        """Read a null-terminated string from byte data"""
        end = data.find(b'\x00', offset)
        return data[offset:end].decode('ascii')
    
    def _read_float(self, data: bytes, offset: int) -> float:
        """Read a 32-bit float from byte data"""
        return struct.unpack('<f', data[offset:offset+4])[0]
    
    def _read_int(self, data: bytes, offset: int) -> int:
        """Read a 32-bit integer from byte data"""
        return struct.unpack('<i', data[offset:offset+4])[0]
    
    def _read_bool(self, data: bytes, offset: int) -> bool:
        """Read a boolean value from byte data"""
        return bool(data[offset])
    
    def _read_position(self, data: bytes, offset: int) -> Position:
        """Read a 3D position from byte data"""
        x = struct.unpack('<f', data[offset:offset+4])[0]
        y = struct.unpack('<f', data[offset+4:offset+8])[0]
        z = struct.unpack('<f', data[offset+8:offset+12])[0]
        return Position(x, y, z)
        
    def _update_string_tables(self, data: bytes) -> None:
        """Update string tables from packet data"""
        try:
            offset = 0
            num_tables = self._read_int(data, offset)
            offset += 4

            for _ in range(num_tables):
                table_id = self._read_int(data, offset)
                offset += 4
                num_entries = self._read_int(data, offset)
                offset += 4

                for _ in range(num_entries):
                    entry_id = self._read_int(data, offset)
                    offset += 4
                    string_length = self._read_int(data, offset)
                    offset += 4
                    string_value = data[offset:offset+string_length].decode('utf-8')
                    offset += string_length

                    # Update the string table with the new entry
                    self._update_string_table(table_id, entry_id, string_value)

        except Exception as e:
            logger.error(f"Error updating string tables: {e}")

    def _update_string_table(self, table_id: int, entry_id: int, string_value: str) -> None:
        """Update a specific string table entry"""
        if table_id not in self.string_tables:
            self.string_tables[table_id] = {}

        self.string_tables[table_id][entry_id] = string_value

    def _update_data_tables(self, data: bytes) -> None:
        """Update data tables from packet data"""
        try:
            offset = 0
            num_tables = self._read_int(data, offset)
            offset += 4

            for _ in range(num_tables):
                table_id = self._read_int(data, offset)
                offset += 4
                num_entries = self._read_int(data, offset)
                offset += 4

                for _ in range(num_entries):
                    entry_id = self._read_int(data, offset)
                    offset += 4
                    entry_type = self._read_int(data, offset)
                    offset += 4
                    entry_size = self._read_int(data, offset)
                    offset += 4
                    entry_data = data[offset:offset+entry_size]
                    offset += entry_size

                    # Update the data table with the new entry
                    self._update_data_table(table_id, entry_id, entry_type, entry_data)

        except Exception as e:
            logger.error(f"Error updating data tables: {e}")
    

    def _update_data_table(self, table_id: int, entry_id: int, entry_type: int, entry_data: bytes) -> None:
        """Update a specific data table entry"""
        if table_id not in self.data_tables:
            self.data_tables[table_id] = {}

        entry_value = self._parse_data_table_entry(entry_type, entry_data)
        if entry_value is not None:
            self.data_tables[table_id][entry_id] = entry_value
    
    def _parse_data_table_entry(self, entry_type: int, entry_data: bytes) -> Union[str, int, float, None]:
        """Parse a data table entry based on the entry type"""
        if entry_type == 0:
            # Parse the entry data as a string
            try:
                return entry_data.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning(f"Error decoding data table entry as string: {entry_data}")
                return None
        elif entry_type == 1:
            # Parse the entry data as an integer
            try:
                return self._read_int(entry_data, 0)
            except struct.error:
                logger.warning(f"Error reading data table entry as integer: {entry_data}")
                return None
        elif entry_type == 2:
            # Parse the entry data as a float
            try:
                return self._read_float(entry_data, 0)
            except struct.error:
                logger.warning(f"Error reading data table entry as float: {entry_data}")
                return None
        else:
            # Unsupported entry type
            logger.warning(f"Unsupported data table entry type: {entry_type}")
            return None