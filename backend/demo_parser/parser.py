from __future__ import annotations
from dataclasses import dataclass, field
from typing import Counter, Dict, List, Optional, Tuple, Set, Any, BinaryIO, Literal, Union, ClassVar
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import struct
import logging
import json
import io
import sys
import os
import re


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class DemoParserException(Exception):
    pass

class DemoParserCorruptedFileException(DemoParserException):
    pass

class DemoMessageTypeHandler:
    def __init__(self, parser: 'DemoParser'):
        self.parser = parser

    def handle_packet(self, packet: DemoPacket) -> None:
        raise NotImplementedError()

class GameEventHandler(DemoMessageTypeHandler):
    def handle_packet(self, packet: DemoPacket) -> None:
        events = self.parser._decode_packet(packet.data)
        for event in events:
            self.parser._handle_event(event)

class StringTableHandler(DemoMessageTypeHandler):
    def handle_packet(self, packet: DemoPacket) -> None:
        self.parser._update_string_tables(packet.data)

class DataTableHandler(DemoMessageTypeHandler):
    def handle_packet(self, packet: DemoPacket) -> None:
        self.parser._update_data_tables(packet.data)

class PacketReadError(Exception):
    pass

class DemoPacketType:
    """CS2 Demo packet types including PBDEMS2 extensions"""
    DEM_STOP = 2
    DEM_FILEHEADER = 3
    DEM_FILEINFO = 4
    DEM_SYNCTICK = 5
    DEM_MESSAGE = 6
    DEM_PACKET = 7
    DEM_SIGNONPACKET = 8
    DEM_CONSOLECMD = 9
    DEM_USERCMD = 10
    DEM_DATATABLES = 11
    DEM_STRINGTABLES = 12
    DEM_USERDATA = 13
    DEM_CUSTOMDATA = 14
    DEM_STRINGCMD = 15
    DEM_SVCMD = 16
    DEM_VOICEDATA = 17
    PBDEMS_CUSTOMDATA = 32

    @classmethod
    def is_valid(cls, cmd_type: int) -> bool:
        """Check if a command type is valid"""
        return cmd_type in {
            cls.DEM_STOP,
            cls.DEM_FILEHEADER,
            cls.DEM_FILEINFO,
            cls.DEM_SYNCTICK,
            cls.DEM_MESSAGE,
            cls.DEM_PACKET,
            cls.DEM_SIGNONPACKET,
            cls.DEM_CONSOLECMD,
            cls.DEM_USERCMD,
            cls.DEM_DATATABLES,
            cls.DEM_STRINGTABLES,
            cls.DEM_USERDATA,
            cls.DEM_CUSTOMDATA,
            cls.DEM_STRINGCMD,
            cls.DEM_SVCMD,
            cls.DEM_VOICEDATA,
            cls.PBDEMS_CUSTOMDATA,
        }

    @classmethod
    def get_name(cls, cmd_type: int) -> str:
        """Get the name of a command type"""
        for name, value in vars(cls).items():
            if not name.startswith('_') and value == cmd_type:
                return name
        return f"UNKNOWN_{cmd_type}"

@dataclass
class DemoPacket:
    cmd_type: int
    tick: int
    data: bytes

@dataclass
class DataTableEntry:
    """Structure for data table entries"""
    entry_id: int
    name: str
    data_type: DataType
    data: Union[int, float, str, Vector3, list, bool]
    table_id: int

    @classmethod
    def from_bytes(cls, data: bytes, offset: int) -> Optional['DataTableEntry']:
        """
        Create DataTableEntry from bytes starting at offset
        
        Args:
            data: Raw bytes containing entry data
            offset: Starting position in bytes
            
        Returns:
            Optional[DataTableEntry]: Parsed entry or None if invalid
        """
        try:
            # Read entry header
            entry_id = struct.unpack('<i', data[offset:offset+4])[0]
            type_id = struct.unpack('<i', data[offset+4:offset+8])[0]
            table_id = struct.unpack('<i', data[offset+8:offset+12])[0]
            name_length = struct.unpack('<i', data[offset+12:offset+16])[0]

            # Validate name length
            if name_length <= 0 or offset + 16 + name_length > len(data):
                logger.warning(f"Invalid name length: {name_length}")
                return None

            # Read name
            name = data[offset+16:offset+16+name_length].decode('utf-8', errors='replace')
            current_offset = offset + 16 + name_length

            # Get data type
            data_type = DataType.from_id(type_id)

            # Parse data based on type
            parsed_data, _ = cls._parse_data_by_type(data, current_offset, data_type)

            return cls(
                entry_id=entry_id,
                name=name,
                data_type=data_type,
                data=parsed_data,
                table_id=table_id
            )

        except Exception as e:
            logger.error(f"Error parsing data table entry: {e}", exc_info=True)
            return None

    @staticmethod
    def _parse_data_by_type(data: bytes, offset: int, data_type: DataType) -> tuple[Any, int]:
        """Parse data of specific type and return (parsed_data, bytes_read)"""
        try:
            if data_type == DataType.INTEGER:
                value = struct.unpack('<i', data[offset:offset+4])[0]
                return value, 4

            elif data_type == DataType.FLOAT:
                value = struct.unpack('<f', data[offset:offset+4])[0]
                return value, 4

            elif data_type == DataType.STRING:
                str_length = struct.unpack('<i', data[offset:offset+4])[0]
                string = data[offset+4:offset+4+str_length].decode('utf-8', errors='replace')
                return string, 4 + str_length

            elif data_type == DataType.VECTOR:
                x = struct.unpack('<f', data[offset:offset+4])[0]
                y = struct.unpack('<f', data[offset+4:offset+8])[0]
                z = struct.unpack('<f', data[offset+8:offset+12])[0]
                return Vector3(x, y, z), 12

            elif data_type == DataType.ARRAY:
                array_length = struct.unpack('<i', data[offset:offset+4])[0]
                array_type = DataType.from_id(struct.unpack('<i', data[offset+4:offset+8])[0])
                
                array_data = []
                current_offset = offset + 8
                bytes_read = 8

                for _ in range(array_length):
                    value, value_bytes = DataTableEntry._parse_data_by_type(
                        data, current_offset, array_type
                    )
                    array_data.append(value)
                    current_offset += value_bytes
                    bytes_read += value_bytes

                return array_data, bytes_read

            elif data_type == DataType.BOOLEAN:
                value = bool(data[offset])
                return value, 1

            else:
                logger.warning(f"Unknown data type: {data_type}")
                return None, 0

        except Exception as e:
            logger.error(f"Error parsing data of type {data_type}: {e}", exc_info=True)
            return None, 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary format"""
        return {
            'entry_id': self.entry_id,
            'name': self.name,
            'type': self.data_type.name,
            'data': self._format_data_for_dict(),
            'table_id': self.table_id
        }

    def _format_data_for_dict(self) -> Any:
        """Format data for dictionary representation"""
        if isinstance(self.data, Vector3):
            return {'x': self.data.x, 'y': self.data.y, 'z': self.data.z}
        return self.data

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (
            f"DataTableEntry(id={self.entry_id}, "
            f"name='{self.name}', "
            f"type={self.data_type.name}, "
            f"data={self.data})"
        )

class DataType(Enum):
    """Supported data types for table entries"""
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    VECTOR = auto()
    ARRAY = auto()
    BOOLEAN = auto()
    UNKNOWN = auto()

    @classmethod
    def from_id(cls, type_id: int) -> 'DataType':
        """Convert numeric type ID to DataType"""
        type_map = {
            0: cls.INTEGER,
            1: cls.FLOAT,
            2: cls.STRING,
            3: cls.VECTOR,
            4: cls.ARRAY,
            5: cls.BOOLEAN
        }
        return type_map.get(type_id, cls.UNKNOWN)

@dataclass
class Vector3:
    """3D Vector representation"""
    x: float
    y: float
    z: float

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }
    
class DemoMessageType(Enum):
    """CS2 Demo message types"""
    DEM_STOP = 2
    DEM_FILEHEADER = 3
    DEM_FILEINFO = 4
    DEM_SYNCTICK = 5
    DEM_MESSAGE = 6
    DEM_PACKET = 7
    DEM_SIGNONPACKET = 8
    DEM_CONSOLECMD = 9
    DEM_USERCMD = 10
    DEM_DATATABLES = 11
    DEM_STRINGTABLES = 12

class ParserError(Exception):
    """Base exception for parser errors"""
    pass

class EventParsingError(ParserError):
    """Exception for errors during event parsing"""
    def __init__(self, event_type: int, message: str, original_error: Optional[Exception] = None):
        self.event_type = event_type
        self.original_error = original_error
        super().__init__(f"Error parsing event type {event_type}: {message}")

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

from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EventType(Enum):
    """CS2 game event types with comprehensive categorization"""
    # Core game events
    UNKNOWN = 0
    GAME_START = 1
    ROUND_START = 2
    ROUND_END = 3
    
    # Player events
    PLAYER_DEATH = 4
    PLAYER_HURT = 5
    PLAYER_SPAWN = 8
    PLAYER_TEAM = 9
    PLAYER_POSITION = 10
    
    # Weapon events
    WEAPON_FIRE = 6
    FOOTSTEP = 7
    
    # Bomb events
    BOMB_PLANTED = 11
    BOMB_DEFUSED = 12
    BOMB_EXPLODED = 13
    
    # Utility events
    GRENADE_THROWN = 14
    DECOY_STARTED = 15
    SMOKE_START = 16
    FLASH_EXPLODE = 17
    HE_DETONATE = 18
    MOLOTOV_DETONATE = 19

    @property
    def is_utility(self) -> bool:
        """Check if event is utility-based"""
        return self in {
            EventType.GRENADE_THROWN,
            EventType.DECOY_STARTED,
            EventType.SMOKE_START,
            EventType.FLASH_EXPLODE,
            EventType.HE_DETONATE,
            EventType.MOLOTOV_DETONATE
        }

    @property
    def is_bomb(self) -> bool:
        """Check if event is bomb-related"""
        return self in {
            EventType.BOMB_PLANTED,
            EventType.BOMB_DEFUSED,
            EventType.BOMB_EXPLODED
        }

    @property
    def is_player(self) -> bool:
        """Check if event is player-related"""
        return self in {
            EventType.PLAYER_DEATH,
            EventType.PLAYER_HURT,
            EventType.PLAYER_SPAWN,
            EventType.PLAYER_TEAM,
            EventType.PLAYER_POSITION
        }

    @property
    def is_round(self) -> bool:
        """Check if event is round-related"""
        return self in {
            EventType.ROUND_START,
            EventType.ROUND_END,
            EventType.GAME_START
        }

    @property
    def is_movement(self) -> bool:
        """Check if event is movement-related"""
        return self in {
            EventType.FOOTSTEP,
            EventType.PLAYER_POSITION
        }

    @property
    def requires_position(self) -> bool:
        """Check if event type typically includes position data"""
        return self in {
            EventType.PLAYER_POSITION,
            EventType.PLAYER_DEATH,
            EventType.BOMB_PLANTED,
            EventType.BOMB_DEFUSED,
            EventType.GRENADE_THROWN,
            EventType.SMOKE_START,
            EventType.FLASH_EXPLODE,
            EventType.HE_DETONATE,
            EventType.MOLOTOV_DETONATE
        }

    @classmethod
    def from_id(cls, event_id: int) -> 'EventType':
        """
        Convert numeric event ID to EventType
        
        Args:
            event_id: Numeric identifier for the event type
            
        Returns:
            EventType: Corresponding event type or UNKNOWN if invalid
        """
        try:
            return cls(event_id)
        except ValueError:
            logger.warning(f"Unknown event type ID: {event_id}")
            return cls.UNKNOWN

    def __str__(self) -> str:
        """Human-readable representation"""
        return self.name

    @property
    def category(self) -> str:
        """
        Get the category of the event
        
        Returns:
            str: Category name ('utility', 'bomb', 'player', 'round', or 'other')
        """
        if self.is_utility:
            return "utility"
        elif self.is_bomb:
            return "bomb"
        elif self.is_player:
            return "player"
        elif self.is_round:
            return "round"
        elif self.is_movement:
            return "movement"
        else:
            return "other"
    
    @property
    def expected_data_fields(self) -> set[str]:
        """
        Get the expected data fields for this event type
        
        Returns:
            set[str]: Set of field names expected in the event data
        """
        if self == EventType.PLAYER_DEATH:
            return {'killer_id', 'victim_id', 'weapon', 'headshot'}
        elif self == EventType.PLAYER_HURT:
            return {'attacker_id', 'victim_id', 'damage', 'hitgroup'}
        elif self == EventType.PLAYER_POSITION:
            return {'player_id', 'position', 'angle', 'velocity'}
        elif self == EventType.BOMB_PLANTED:
            return {'player_id', 'site', 'position'}
        elif self == EventType.BOMB_DEFUSED:
            return {'player_id', 'site'}
        elif self.is_utility:
            return {'player_id', 'position', 'velocity'}
        else:
            return set()

    def validate_event_data(self, data: dict) -> bool:
        """
        Validate that event data contains required fields
        
        Args:
            data: Event data dictionary to validate
            
        Returns:
            bool: True if data contains all required fields
        """
        required_fields = self.expected_data_fields
        return all(field in data for field in required_fields)

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
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the event data dictionary, with a default value"""
        return self.data.get(key, default)

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

@dataclass
class DemoHeader:
    """CS2 Demo file header"""
    
    # Class constants
    HEADER_SIZE: ClassVar[int] = 1072
    SUPPORTED_FORMATS = ["HL2DEMO", "PBDEMS2"]
    
    # Fields with default values
    raw_data: bytes = field(repr=False)  # Exclude from repr to avoid clutter
    magic: str = field(default="")
    demo_protocol: int = field(default=0)
    network_protocol: int = field(default=0)
    server_name: str = field(default="")
    client_name: str = field(default="")
    map_name: str = field(default="")
    game_directory: str = field(default="")
    playback_time: float = field(default=0.0)
    ticks: int = field(default=0)
    frames: int = field(default=0)
    signon_length: int = field(default=0)

    def __post_init__(self):
        """Validate header data after initialization"""
        if len(self.raw_data) != self.HEADER_SIZE:
            raise ValueError(
                f"Invalid header size: {len(self.raw_data)} bytes, "
                f"expected {self.HEADER_SIZE} bytes"
            )
        
        # Check for supported magic strings
        magic_upper = self.magic.upper().strip('\0')
        if not any(magic_upper.startswith(fmt) for fmt in self.SUPPORTED_FORMATS):
            raise ValueError(
                f"Invalid demo file magic: {self.magic}. "
                f"Expected one of: {', '.join(self.SUPPORTED_FORMATS)}"
            )
            
        # Validate numeric fields
        if self.ticks < 0:
            raise ValueError(f"Invalid tick count: {self.ticks}")
        if self.frames < 0:
            raise ValueError(f"Invalid frame count: {self.frames}")
        if self.playback_time < 0:
            raise ValueError(f"Invalid playback time: {self.playback_time}")

    @classmethod
    def from_bytes(cls, raw_data: bytes) -> 'DemoHeader':
        """
        Create a DemoHeader instance from raw bytes.
        
        Args:
            raw_data: Raw header bytes (must be 1072 bytes)
            
        Returns:
            DemoHeader: Parsed header instance
            
        Raises:
            ValueError: If header size is invalid or magic string doesn't match
        """
        if len(raw_data) != cls.HEADER_SIZE:
            raise ValueError(
                f"Invalid header size: {len(raw_data)} bytes, "
                f"expected {cls.HEADER_SIZE} bytes"
            )

        try:
            # Try to detect format first
            magic = raw_data[:8].decode('ascii', errors='replace').strip('\0')
            magic_upper = magic.upper()
            
            # Log the magic string for debugging
            logger.debug(f"Detected magic string: {magic}")
            
            header = cls(
                raw_data=raw_data,
                magic=magic,
                demo_protocol=struct.unpack('i', raw_data[8:12])[0],
                network_protocol=struct.unpack('i', raw_data[12:16])[0],
                server_name=raw_data[16:272].decode('ascii', errors='replace').strip('\0'),
                client_name=raw_data[272:528].decode('ascii', errors='replace').strip('\0'),
                map_name=raw_data[528:784].decode('ascii', errors='replace').strip('\0'),
                game_directory=raw_data[784:1040].decode('ascii', errors='replace').strip('\0'),
                playback_time=struct.unpack('f', raw_data[1040:1044])[0],
                ticks=struct.unpack('i', raw_data[1044:1048])[0],
                frames=struct.unpack('i', raw_data[1048:1052])[0],
                signon_length=struct.unpack('i', raw_data[1052:1056])[0]
            )
            
            logger.debug(f"Successfully created header: {header}")
            return header
            
        except (struct.error, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to parse demo header: {str(e)}") from e
    def to_dict(self) -> Dict[str, Any]:
        """Convert header to a dictionary format for serialization."""
        return {
            'magic': self.magic,
            'demo_protocol': self.demo_protocol,
            'network_protocol': self.network_protocol,
            'server_name': self.server_name,
            'client_name': self.client_name,
            'map_name': self.map_name,
            'game_directory': self.game_directory,
            'playback_time': self.playback_time,
            'ticks': self.ticks,
            'frames': self.frames,
            'signon_length': self.signon_length
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"CS2 Demo: {self.map_name}\n"
            f"Format: {self.magic}\n"
            f"Recorded by: {self.client_name}\n"
            f"Server: {self.server_name}\n"
            f"Duration: {self.playback_time:.2f}s\n"
            f"Ticks: {self.ticks}"
        )

    @classmethod
    def from_file(cls, file_path: str | Path) -> 'DemoHeader':
        """
        Create a DemoHeader instance directly from a demo file.
        
        Args:
            file_path: Path to the demo file
            
        Returns:
            DemoHeader: Parsed header instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If header is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Demo file not found: {path}")
            
        try:
            with path.open('rb') as f:
                header_data = f.read(cls.HEADER_SIZE)
                return cls.from_bytes(header_data)
        except Exception as e:
            raise ValueError(f"Failed to read demo file: {str(e)}") from e
class DemoParser:
    """CS2 Demo Parser with comprehensive analysis capabilities"""

    HEADER_SIZE = 1072  # Updated correct header size
    TICK_RATE = 64
    PLAYERS_PER_TEAM = 5
    MAX_PACKET_SIZE = 1024 * 1024 
    
    def _read_data_table_entry(self, data: bytes, offset: int, table_id: int) -> Optional[DataTableEntry]:
        """
        Read a single data table entry from the given offset
        
        Args:1
            data: Raw bytes containing the entry
            offset: Starting position to read from
            table_id: ID of the current table being processed
            
        Returns:
            Optional[DataTableEntry]: Parsed entry or None if invalid
        """
        try:
            # Validate remaining data length
            if offset + 12 >= len(data):
                logger.warning("Insufficient data length for table entry header")
                return None

            # Read entry header
            entry_id = int.from_bytes(data[offset:offset+4], byteorder='little')
            type_id = int.from_bytes(data[offset+4:offset+8], byteorder='little')
            name_length = int.from_bytes(data[offset+8:offset+12], byteorder='little')

            # Validate name length
            if name_length <= 0 or offset + 12 + name_length > len(data):
                logger.warning(f"Invalid name length: {name_length}")
                return None

            # Read entry name
            name = data[offset+12:offset+12+name_length].decode('utf-8', errors='replace')
            data_start = offset + 12 + name_length

            # Parse data based on type
            try:
                data_type = DataType(type_id)
            except ValueError:
                logger.warning(f"Unknown data type ID: {type_id}")
                data_type = DataType.UNKNOWN

            parsed_data = self._parse_data_value(data[data_start:], data_type)
            
            return DataTableEntry(
                entry_id=entry_id,
                table_id=table_id,  # Added table_id
                name=name,
                data_type=data_type,
                data=parsed_data
            )

        except Exception as e:
            logger.error(f"Error reading data table entry at offset {offset}: {e}", exc_info=True)
            return None
        
    def _handle_command_packet(self, data: bytes) -> None:
        """Handle PBDEMS command packet"""
        if len(data) < 4:
            return
            
        try:
            cmd_type = data[0]
            logger.debug(f"Processing command packet type {cmd_type}")
            # Add command packet handling here
            pass
        except Exception as e:
            logger.error(f"Error handling command packet: {e}")

        def _handle_data_packet(self, data: bytes) -> None:
            """Handle PBDEMS data packet"""
            if len(data) < 4:
                return
                
            try:
                packet_type = data[0]
                logger.debug(f"Processing data packet type {packet_type}")
                # Add data packet handling here
                pass
            except Exception as e:
                logger.error(f"Error handling data packet: {e}")

    def _handle_data_tables(self, data: bytes) -> None:
        """Process data tables from the demo file"""
        try:
            offset = 0
            while offset + 8 < len(data):
                # Read table header
                table_id = int.from_bytes(data[offset:offset+4], byteorder='little')
                num_entries = int.from_bytes(data[offset+4:offset+8], byteorder='little')
                offset += 8
                
                logger.debug(f"Processing data table {table_id} with {num_entries} entries")
                
                # Process entries
                for _ in range(num_entries):
                    entry = self._read_data_table_entry(data, offset, table_id)  # Pass table_id
                    if entry:
                        if table_id not in self.data_tables:
                            self.data_tables[table_id] = {}
                        self.data_tables[table_id][entry.entry_id] = entry
                        offset += self._calculate_entry_size(entry)
                    else:
                        break
                        
        except Exception as e:
            logger.error("Error processing data tables", exc_info=True)

    def _parse_data_value(self, data: bytes, data_type: DataType) -> Any:
        """Parse a value based on its data type"""
        try:
            if data_type == DataType.INTEGER:
                return int.from_bytes(data[:4], byteorder='little')
                
            elif data_type == DataType.FLOAT:
                return struct.unpack('<f', data[:4])[0]
                
            elif data_type == DataType.STRING:
                str_length = int.from_bytes(data[:4], byteorder='little')
                return data[4:4+str_length].decode('utf-8', errors='replace')
                
            elif data_type == DataType.VECTOR:
                x = struct.unpack('<f', data[0:4])[0]
                y = struct.unpack('<f', data[4:8])[0]
                z = struct.unpack('<f', data[8:12])[0]
                return Vector3(x, y, z)
                
            elif data_type == DataType.BOOLEAN:
                return bool(data[0])
                
            elif data_type == DataType.ARRAY:
                length = int.from_bytes(data[:4], byteorder='little')
                element_type = DataType(int.from_bytes(data[4:8], byteorder='little'))
                array = []
                offset = 8
                for _ in range(length):
                    value = self._parse_data_value(data[offset:], element_type)
                    array.append(value)
                    offset += self._get_type_size(element_type)
                return array
                
            else:
                logger.warning(f"Unsupported data type: {data_type}")
                return None

        except Exception as e:
            logger.error(f"Error parsing data of type {data_type}: {e}", exc_info=True)
            return None

    def _calculate_entry_size(self, entry: DataTableEntry) -> int:
        """Calculate the total size of an entry in bytes"""
        base_size = 12 + len(entry.name.encode('utf-8'))  # Header + name
        data_size = self._get_type_size(entry.data_type)
        
        if entry.data_type == DataType.STRING and isinstance(entry.data, str):
            data_size += len(entry.data.encode('utf-8'))
        elif entry.data_type == DataType.ARRAY and isinstance(entry.data, list):
            for item in entry.data:
                if isinstance(item, str):
                    data_size += 4 + len(item.encode('utf-8'))
                else:
                    data_size += self._get_type_size(entry.data_type)
                    
        return base_size + data_size

    def _get_type_size(self, data_type: DataType) -> int:
        """Get the size in bytes for a given data type"""
        return {
            DataType.INTEGER: 4,
            DataType.FLOAT: 4,
            DataType.BOOLEAN: 1,
            DataType.VECTOR: 12,
            DataType.STRING: 4,  # Just the length field, actual string size varies
            DataType.ARRAY: 8,   # Length and element type fields
            DataType.UNKNOWN: 0
        }[data_type]
    
    def __init__(self, demo_path: str, skip_corrupted: bool = True):
        self.demo_path = os.path.abspath(demo_path)
        if not os.path.exists(self.demo_path):
            raise FileNotFoundError(f"Demo file not found: {demo_path}")

        self.header: Optional[DemoHeader] = None
        self.rounds: List[Round] = []
        self.players: Dict[int, PlayerState] = {}
        self.current_round: int = 0
        self.events: List[GameEvent] = []
        self.string_tables: Dict[int, Dict[int, str]] = {}
        self.data_tables: Dict[int, Dict[int, Any]] = {}
        self.skip_corrupted = skip_corrupted

        self._retake_cache: Dict[int, bool] = {}
        self._position_cache: Dict[Tuple[float, float, float], bool] = {}

        self._message_type_handlers: Dict[int, DemoMessageTypeHandler] = {
            DemoMessageType.DEM_PACKET.value: GameEventHandler(self),
            DemoMessageType.DEM_STRINGTABLES.value: StringTableHandler(self),
            DemoMessageType.DEM_DATATABLES.value: DataTableHandler(self),
        }

        logger.info(f"Initialized parser for {demo_path}")
    
    def _validate_demo_file(self):
        """Validate that the file is a valid CS2 demo"""
        try:
            with open(self.demo_path, 'rb') as f:
                magic = f.read(8).decode('ascii').strip('\0')
                if magic != "HL2DEMO":
                    raise ValueError(f"Invalid demo file magic: {magic}, expected 'HL2DEMO'")
        except Exception as e:
            raise ValueError(f"Invalid demo file: {e}")
    
    def _read_header(self, demo_file: BinaryIO) -> None:
        """Read and parse demo header with improved format detection"""
        try:
            header_data = demo_file.read(DemoHeader.HEADER_SIZE)
            if len(header_data) < DemoHeader.HEADER_SIZE:
                raise ValueError(f"Incomplete header: got {len(header_data)} bytes")

            # Try to detect format
            magic_bytes = header_data[:8]
            if magic_bytes.startswith(b"HL2DEMO\0"):
                logger.info("Detected HL2DEMO format")
                self.header = DemoHeader.from_bytes(header_data)
            elif magic_bytes.startswith(b"PBDEMS"):
                logger.info("Detected PBDEMS format")
                self._parse_pbdems2_header(header_data)
            else:
                # Try to decode magic for logging
                magic = magic_bytes.decode('ascii', errors='replace').strip('\0')
                logger.warning(f"Unknown demo format: {magic}")
                # Try parsing as HL2DEMO as fallback
                self.header = DemoHeader.from_bytes(header_data)

        except ValueError as e:
            if self.skip_corrupted:
                logger.info(f"Skipping demo file with parsing error: {str(e)}")
            else:
                raise
    
    def _handle_unknown_format(self, header_data: bytes) -> None:
        """Handle an unknown demo file format"""
        logger.warning(f"Demo file format is not recognized for file: {self.demo_path}")
        if self.skip_corrupted:
            logger.info(f"Skipping demo file with unknown format: {self.demo_path}")
        else:
            logger.warning("Would you like to:")
            logger.warning("1. Skip this demo file and continue parsing?")
            logger.warning("2. Abort the parsing process?")

            user_input = input("Enter 1 to skip, 2 to abort: ")
            if user_input == "1":
                logger.info(f"Skipping demo file with unknown format: {self.demo_path}")
            elif user_input == "2":
                raise DemoParserException("Parsing aborted due to unknown demo file format.")
            else:
                logger.error("Invalid input, aborting parsing.")
                raise DemoParserException("Parsing aborted due to invalid user input.")
    
    def _parse_pbdems2_header(self, header_data: bytes) -> None:
        """Parse PBDEMS2 header with improved string handling"""
        try:
            # Print hex dump for debugging
            logger.info("Header dump (first 128 bytes):")
            for i in range(0, min(128, len(header_data)), 16):
                chunk = header_data[i:i+16]
                hex_str = ' '.join(f'{b:02x}' for b in chunk)
                ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
                logger.info(f"{i:04x}: {hex_str:48s} {ascii_str}")

            # Parse magic string (first 8 bytes)
            magic = header_data[:8].decode('ascii', errors='replace').strip('\0')
            if not magic.startswith('PBDEMS'):
                raise ValueError(f"Invalid PBDEMS magic: {magic}")

            # Parse protocols correctly based on observed format
            demo_protocol = struct.unpack('<I', header_data[8:12])[0]
            network_protocol = struct.unpack('<I', header_data[12:16])[0]
            
            logger.info(f"Demo Protocol: {demo_protocol}")
            logger.info(f"Network Protocol: {network_protocol}")

            def safe_extract_string(start_marker: bytes, max_length: int = 64) -> str:
                """Safely extract a null-terminated string starting with a marker"""
                try:
                    start_idx = header_data.find(start_marker)
                    if start_idx == -1:
                        return ""
                    
                    # Find next null terminator or control character
                    end_idx = start_idx
                    for i in range(start_idx, min(start_idx + max_length, len(header_data))):
                        if header_data[i] < 32:  # Control character or null
                            end_idx = i
                            break
                    
                    # Extract and clean the string
                    string_bytes = header_data[start_idx:end_idx]
                    return string_bytes.decode('ascii', errors='replace').strip()
                except Exception as e:
                    logger.warning(f"Error extracting string with marker {start_marker}: {e}")
                    return ""

            # Extract map name (starts with "de_")
            map_name = safe_extract_string(b'de_', 32)
            # Clean up map name - only take up to first non-alphanumeric character
            map_name = ''.join(c for c in map_name if c.isalnum() or c == '_')
            logger.info(f"Found map name: {map_name}")

            # Extract server name (starts with "FACEIT")
            server_name = safe_extract_string(b'FACEIT', 128)
            logger.info(f"Found server name: {server_name}")

            # Extract game directory (starts with "/home" or "/cs2")
            game_directory = ""
            for path_start in [b'/home', b'/cs2']:
                temp_dir = safe_extract_string(path_start, 256)
                if temp_dir:
                    game_directory = temp_dir
                    break
            logger.info(f"Found game directory: {game_directory}")

            # Set reasonable defaults for timing values
            playback_time = 0.0
            ticks = 0
            frames = 0
            signon_length = 0

            # Create header object with parsed values
            self.header = DemoHeader(
                raw_data=header_data,
                magic=magic,
                demo_protocol=demo_protocol,
                network_protocol=network_protocol,
                server_name=server_name,
                client_name="",  # We'll leave this blank for now
                map_name=map_name,
                game_directory=game_directory,
                playback_time=playback_time,
                ticks=ticks,
                frames=frames,
                signon_length=signon_length
            )

            logger.info("Successfully created header object")
            logger.info(f"Final header values:")
            logger.info(f"Magic: {self.header.magic}")
            logger.info(f"Demo Protocol: {self.header.demo_protocol}")
            logger.info(f"Network Protocol: {self.header.network_protocol}")
            logger.info(f"Map: {self.header.map_name}")
            logger.info(f"Server: {self.header.server_name}")
            logger.info(f"Game Directory: {self.header.game_directory}")

        except Exception as e:
            logger.error(f"Error parsing PBDEMS header: {e}", exc_info=True)
            raise

    def _translate_game_events(self, data: bytes, tick: int) -> None:
        """Translate the game event data"""
        events = self._decode_packet(data)
        for event in events:
            event_type = EventType[event['type']]
            logger.info(f"Event Type: {event_type.name}")
            logger.info(f"Tick: {event['tick']}")
            logger.info(f"Data: {event['data']}")

    def _translate_string_tables(self, data: bytes) -> None:
        """Translate the string table data"""
        try:
            offset = 0
            num_tables = self._read_int(data, offset)
            offset += 4

            for _ in range(num_tables):
                table_id = self._read_int(data, offset)
                offset += 4
                num_entries = self._read_int(data, offset)
                offset += 4

                logger.info(f"String Table ID: {table_id}")
                logger.info(f"Number of Entries: {num_entries}")

                for _ in range(num_entries):
                    entry_id = self._read_int(data, offset)
                    offset += 4
                    string_length = self._read_int(data, offset)
                    offset += 4
                    string_value = data[offset:offset+string_length].decode('utf-8', errors='replace')
                    offset += string_length

                    logger.info(f"Entry ID: {entry_id}")
                    logger.info(f"String Value: {string_value}")

        except Exception as e:
            logger.error(f"Error translating string tables: {e}", exc_info=True)

    def _translate_data_tables(self, data: bytes) -> None:
        """Translate the data table data"""
        try:
            offset = 0
            num_tables = self._read_int(data, offset)
            offset += 4

            for _ in range(num_tables):
                table_id = self._read_int(data, offset)
                offset += 4
                num_entries = self._read_int(data, offset)
                offset += 4

                logger.info(f"Data Table ID: {table_id}")
                logger.info(f"Number of Entries: {num_entries}")

                for _ in range(num_entries):
                    entry_id = self._read_int(data, offset)
                    offset += 4
                    entry_type = self._read_int(data, offset)
                    offset += 4
                    entry_size = self._read_int(data, offset)
                    offset += 4
                    entry_data = data[offset:offset+entry_size]
                    offset += entry_size

                    logger.info(f"Entry ID: {entry_id}")
                    logger.info(f"Entry Type: {entry_type}")
                    logger.info(f"Entry Size: {entry_size}")

                    # Translate the entry data based on the type
                    entry_value = self._translate_data_table_entry(entry_type, entry_data)
                    logger.info(f"Entry Value: {entry_value}")

        except Exception as e:
            logger.error(f"Error translating data tables: {e}", exc_info=True)

    def _translate_data_table_entry(self, entry_type: int, entry_data: bytes) -> str:
        """Translate a data table entry based on the entry type"""
        try:
            if entry_type == 0:
                # Parse the entry data as a string
                return entry_data.decode('utf-8', errors='replace')
            elif entry_type == 1:
                # Parse the entry data as an integer
                return str(self._read_int(entry_data, 0))
            elif entry_type == 2:
                # Parse the entry data as a float
                return str(self._read_float(entry_data, 0))
            else:
                # Unsupported entry type
                return f"Unknown type: {entry_type}"
        except Exception as e:
            logger.warning(f"Error translating data table entry: {e}")
            return "Error"


    def _read_message(self, demo_file: BinaryIO) -> Optional[Tuple[DemoMessageType, bytes]]:
        """Read the next message from the demo file"""
        try:
            # Read message type
            type_data = demo_file.read(1)
            if not type_data:
                return None
                
            message_type = DemoMessageType(int.from_bytes(type_data, byteorder='little'))
            
            # Read tick number
            tick = int.from_bytes(demo_file.read(4), byteorder='little')
            self.current_tick = tick
            
            # Read message slot (unused in CS2)
            slot = int.from_bytes(demo_file.read(1), byteorder='little')
            
            # Read message data length
            length = int.from_bytes(demo_file.read(4), byteorder='little')
            
            # Read message data
            data = demo_file.read(length)
            if len(data) < length:
                logger.warning(f"Incomplete message data: expected {length} bytes, got {len(data)}")
                return None
                
            return message_type, data
            
        except EOFError:
            return None
        except Exception as e:
            logger.error(f"Error reading message: {e}")
            return None

    def _parse_demo(self) -> None:
        with open(self.demo_path, 'rb') as demo_file:
            self._read_header(demo_file)
            while True:
                packet = self._read_next_packet(demo_file)
                if not packet:
                    break
                self._process_packet(packet)


    def _parse_demo_header(self, demo_file: BinaryIO) -> Optional[DemoHeader]:
        """Parse the demo file header"""
        try:
            # Read the header fields
            raw_data = demo_file.read(DemoHeader.HEADER_SIZE)
            if len(raw_data) != DemoHeader.HEADER_SIZE:
                logger.warning(f"Invalid header size: {len(raw_data)} bytes")
                return None

            magic = raw_data[:8].decode('utf-8').rstrip('\0')
            if magic != "HL2DEMO":
                logger.warning(f"Invalid demo file magic: {magic}")
                return None

            demo_protocol = self._read_int(raw_data[8:12], 0)
            network_protocol = self._read_int(raw_data[12:16], 0)
            server_name = raw_data[16:276].decode('utf-8').rstrip('\0')
            client_name = raw_data[276:536].decode('utf-8').rstrip('\0')
            map_name = raw_data[536:796].decode('utf-8').rstrip('\0')
            game_directory = raw_data[796:1056].decode('utf-8').rstrip('\0')
            playback_time = self._read_float(raw_data[1056:1060], 0)
            ticks = self._read_int(raw_data[1060:1064], 0)
            frames = self._read_int(raw_data[1064:1068], 0)
            signon_length = self._read_int(raw_data[1068:1072], 0)

            # Create the DemoHeader instance
            header = DemoHeader(
                raw_data=raw_data,
                magic=magic,
                demo_protocol=demo_protocol,
                network_protocol=network_protocol,
                server_name=server_name,
                client_name=client_name,
                map_name=map_name,
                game_directory=game_directory,
                playback_time=playback_time,
                ticks=ticks,
                frames=frames,
                signon_length=signon_length
            )

            logger.debug(f"Parsed demo header: {header}")
            return header

        except Exception as e:
            logger.error(f"Error parsing demo header: {e}", exc_info=True)
            return None
        
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

    def _analyze_file_structure(self, demo_file: BinaryIO) -> None:
        """Analyze overall file structure"""
        # Save current position
        current_pos = demo_file.tell()

        try:
            # Get file size
            demo_file.seek(0, 2)  # Seek to end
            file_size = demo_file.tell()
            demo_file.seek(current_pos)  # Return to original position

            # Read and analyze chunks of file
            chunk_size = 256
            position = current_pos
            
            logger.info(f"Analyzing file structure from position {current_pos} (file size: {file_size})")
            
            while position < file_size and position < current_pos + 1024:  # Look at first 1KB
                demo_file.seek(position)
                chunk = demo_file.read(min(chunk_size, file_size - position))
                
                # Print chunk info
                logger.info(f"\nChunk at position {position}:")
                for i in range(0, len(chunk), 16):
                    hex_data = chunk[i:i+16]
                    # Hex view
                    hex_str = ' '.join(f'{b:02x}' for b in hex_data)
                    # ASCII view
                    ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in hex_data)
                    # Offset and data
                    logger.info(f"{position+i:08x}: {hex_str:48s} {ascii_str}")
                
                position += chunk_size

        finally:
            # Restore original position
            demo_file.seek(current_pos)
    
    
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
    
    def _save_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save the analysis results"""
        try:
            # Construct the output file path
            output_dir = os.path.join(os.path.dirname(self.demo_path), 'analysis')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.demo_path))[0]}.json")

            # Save the analysis data to a JSON file
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=4)

            logger.info(f"Analysis saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")

    def _resync_to_valid_packet(self, demo_file: BinaryIO) -> Optional[Tuple[int, int, int]]:
        """Attempt to resynchronize to the next valid packet"""
        MAX_SCAN_BYTES = 1024 * 1024  # 1MB
        VALID_TYPES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32}
        bytes_scanned = 0
        buffer = bytearray()

        while bytes_scanned < MAX_SCAN_BYTES:
            byte = demo_file.read(1)
            if not byte:
                return None  # EOF
            bytes_scanned += 1
            buffer.extend(byte)

            if len(buffer) >= 9:  # Minimum packet header size
                if self._looks_like_packet_start(buffer[-9:]):
                    # Found potential packet start
                    new_pos = demo_file.tell() - 9
                    demo_file.seek(new_pos)
                    logger.info(f"Found potential packet at offset {new_pos - bytes_scanned} from scan start")
                    header = self._read_packet_header(demo_file)
                    if header:
                        cmd_type, tick, size = header
                        if cmd_type in VALID_TYPES:
                            return cmd_type, tick, size
                # Keep last 8 bytes for next iteration
                buffer = buffer[-8:]

        logger.warning(f"Could not find valid packet in {bytes_scanned} bytes")
        return None
    
    def _read_packet(self, demo_file: BinaryIO) -> Optional[DemoPacket]:
        """Read a standard data packet from the demo file"""
        try:
            # Read the packet header
            header_bytes = demo_file.read(4)
            packet_size = self._read_int(header_bytes, 0)  # Use 0 as the offset

            if packet_size <= 0:
                logger.warning(f"Invalid packet size: {packet_size}")
                return None

            packet_data = demo_file.read(packet_size)
            if len(packet_data) < packet_size:
                logger.warning(f"Incomplete packet data: expected {packet_size}, got {len(packet_data)}")
                return None

            # Determine the packet type from the first byte
            packet_type = packet_data[0]

            # Create the DemoPacket instance
            return DemoPacket(
                cmd_type=packet_type,
                tick=0,  # Tick information is not always available in the header
                data=packet_data
            )

        except Exception as e:
            logger.error(f"Error reading packet: {e}", exc_info=True)
            return None

        
    def _read_bytes(self, demo_file: BinaryIO, size: int) -> bytes:
        """Reads a specified number of bytes from the demo file."""
        return demo_file.read(size)

        
    def _looks_like_packet_start(self, header_bytes: Union[bytes, bytearray]) -> bool:
        """Check if bytes look like a valid packet header"""
        if len(header_bytes) < 9:
            return False

        cmd_type = header_bytes[0]
        tick = int.from_bytes(header_bytes[1:5], 'little')
        size = int.from_bytes(header_bytes[5:9], 'little')

        # Known good patterns for PBDEMS2
        KNOWN_TYPES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32}

        # Command type must be valid
        if cmd_type not in KNOWN_TYPES:
            return False

        # Size must be reasonable
        MAX_REASONABLE_SIZE = 10 * 1024 * 1024  # 10MB absolute max
        if size <= 0 or size > MAX_REASONABLE_SIZE:
            return False

        # Tick should be reasonable for most packet types
        if cmd_type not in {3, 4, 9}:  # Exclude special packets that might have special tick values
            if tick < 0 or tick > 1000000:  # 1M ticks max
                return False

        return True

    def _is_reasonable_size(self, size: int, cmd_type: int) -> bool:
        """Check if packet size is reasonable for the given type"""
        if size <= 0:
            return False

        # Different size limits for different packet types
        if cmd_type == 32:  # PBDEMS_CUSTOMDATA
            return size <= 10 * 1024 * 1024  # 10MB
        elif cmd_type in {11, 12}:  # Tables
            return size <= 5 * 1024 * 1024   # 5MB
        else:
            return size <= 1024 * 1024        # 1MB
    
    def _is_valid_packet(self, cmd_type: int, tick: int, size: int) -> bool:
        """Validate packet header values"""
        # Define valid packet types
        VALID_TYPES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32}
        
        # Type validation
        if cmd_type not in VALID_TYPES:
            return False
            
        # Tick validation - some packet types can have special tick values
        if cmd_type not in {3, 4, 9}:  # Normal packets
            if tick < 0 or tick > 1_000_000:  # Reasonable tick range
                return False
                
        # Size validation - different limits for different types
        if size <= 0:
            return False
            
        if cmd_type in {11, 12}:  # Table packets
            MAX_SIZE = 5 * 1024 * 1024  # 5MB
        elif cmd_type == 32:  # Custom data
            MAX_SIZE = 10 * 1024 * 1024  # 10MB
        else:  # Regular packets
            MAX_SIZE = 1024 * 1024  # 1MB
            
        if size > MAX_SIZE:
            return False
            
        return True

    def _read_next_packet(self, demo_file: BinaryIO) -> Optional[DemoPacket]:
        """Read next packet with enhanced debugging"""
        # Initialize position
        current_pos: int = demo_file.tell()
        
        try:
            # First, let's look at a larger chunk of data
            preview_size = 64  # Look at 64 bytes at a time
            preview_data = demo_file.read(preview_size)
            demo_file.seek(current_pos)  # Reset position
            
            # Print hex dump of data
            logger.info(f"\nReading at position {current_pos}:")
            logger.info("Preview of next 64 bytes:")
            hex_dump = ' '.join(f'{b:02x}' for b in preview_data)
            ascii_dump = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in preview_data)
            logger.info(f"HEX:   {hex_dump}")
            logger.info(f"ASCII: {ascii_dump}")
            
            # Look for common patterns
            patterns = {
                'PBDEMS Header': b'PBDEMS',
                'Command Start': b'\x07\xd0',
                'Data Start': b'\x01\xf1',
                'File Info': b'\x04\x00',
                'String Table': b'\x0c\x00',
                'Data Table': b'\x0b\x00',
            }
            
            logger.info("\nSearching for known patterns:")
            for name, pattern in patterns.items():
                pos = preview_data.find(pattern)
                if pos != -1:
                    logger.info(f"Found {name} at offset +{pos}")
                    
            # Try to determine packet structure
            if len(preview_data) >= 4:
                possible_sizes = []
                for i in range(len(preview_data)-3):
                    size = int.from_bytes(preview_data[i:i+4], 'little')
                    if 0 < size < 1024*1024:  # Reasonable size
                        possible_sizes.append((i, size))
                
                if possible_sizes:
                    logger.info("\nPossible packet sizes found:")
                    for offset, size in possible_sizes:
                        logger.info(f"Offset +{offset}: size = {size}")

            # Try to parse based on what looks like packet headers
            # First, skip any non-packet data
            skip_size = 0
            for i in range(min(32, len(preview_data))):
                # Look for packet header patterns
                if (i + 1 < len(preview_data) and 
                    # Check for known packet type bytes
                    (preview_data[i] in {0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C} and
                    preview_data[i+1] == 0x00)):
                    skip_size = i
                    break
                    
                # Check for PBDEMS2 data marker
                if (i + 3 < len(preview_data) and 
                    preview_data[i:i+2] in {b'\x07\xd0', b'\x01\xf1'}):
                    skip_size = i
                    break
            
            if skip_size > 0:
                logger.info(f"\nSkipping {skip_size} bytes to potential packet start")
                demo_file.seek(current_pos + skip_size)
                
            # Try to read packet header
            header_bytes = demo_file.read(4)
            if len(header_bytes) < 4:
                logger.info("Not enough bytes for header")
                return None
                
            logger.info(f"\nPotential packet header: {' '.join(f'{b:02x}' for b in header_bytes)}")
            
            # Basic packet structure
            if header_bytes[0] in {0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C}:
                # Standard packet
                packet_type = header_bytes[0]
                size = int.from_bytes(header_bytes[1:4], 'little')
                
                logger.info(f"Standard packet: type={packet_type}, size={size}")
                
                if 0 < size < 1024*1024:  # Reasonable size
                    data = demo_file.read(size)
                    return DemoPacket(cmd_type=packet_type, tick=0, data=data)
                    
            elif header_bytes[0:2] in {b'\x07\xd0', b'\x01\xf1'}:
                # PBDEMS style packet
                packet_type = header_bytes[2]
                size = header_bytes[3]
                
                logger.info(f"PBDEMS packet: type={packet_type}, size={size}")
                
                if 0 < size < 1024*1024:  # Reasonable size
                    data = demo_file.read(size)
                    return DemoPacket(cmd_type=packet_type, tick=0, data=data)

            # If we get here, try advancing by 1 and looking again
            demo_file.seek(current_pos + 1)
            return None

        except Exception as e:
            logger.error(f"Error in packet reading at position {current_pos}: {e}")
            demo_file.seek(current_pos + 1)
            return None
        
    def _peek_bytes(self, demo_file: BinaryIO, num_bytes: int) -> bytes:
        """Peek at the next bytes without moving file pointer"""
        pos = demo_file.tell()
        data = demo_file.read(num_bytes)
        demo_file.seek(pos)
        return data
        
    def _find_next_packet(self, demo_file: BinaryIO) -> Optional[DemoPacket]:
        """Find the next valid packet in the file"""
        MAX_SCAN_BYTES = 1024 * 1024  # Max 1MB scan
        start_pos = demo_file.tell()
        bytes_scanned = 0
        buffer = bytearray()

        while bytes_scanned < MAX_SCAN_BYTES:
            byte = demo_file.read(1)
            if not byte:
                return None  # EOF

            bytes_scanned += 1
            buffer.extend(byte)

            if len(buffer) >= 9:  # Minimum packet header size
                if self._looks_like_packet_start(bytes(buffer[-9:])):
                    # Found potential packet start
                    new_pos = demo_file.tell() - 9
                    demo_file.seek(new_pos)
                    logger.info(f"Found potential packet at offset {new_pos - start_pos} from scan start")
                    header = self._read_packet_header(demo_file)
                    if header:
                        cmd_type, tick, size = header
                        if self._is_valid_packet_header(cmd_type, tick, size):
                            return DemoPacket(cmd_type=cmd_type, tick=tick, data=self._read_packet_data(demo_file, size))

                # Keep last 8 bytes for next iteration
                buffer = buffer[-8:]

        logger.warning(f"Could not find valid packet in {bytes_scanned} bytes")
        return None
        
    def _validate_packet_header(self, cmd_type: int, tick: int, size: int) -> bool:
        """Validate packet header values with PBDEMS2 support"""
        # Basic size validation
        if size < 0:
            logger.warning(f"Negative packet size: {size}")
            return False

        # Different size limits for different packet types
        if cmd_type == 32:  # PBDEMS_CUSTOMDATA
            MAX_SIZE = 10 * 1024 * 1024  # 10MB
        else:
            MAX_SIZE = 1024 * 1024  # 1MB

        if size > MAX_SIZE:
            logger.warning(f"Packet size too large: {size} > {MAX_SIZE}")
            return False

        # Tick validation - allow negative ticks for special packets
        if cmd_type in {3, 4, 9}:  # Header/info/command packets
            return True
        if tick < 0 or tick > 1000000:
            logger.warning(f"Invalid tick number: {tick}")
            return False

        return True
        
    def _read_packet_data(self, demo_file: BinaryIO, size: int) -> bytes:
        """Read packet data with improved error handling"""
        if size <= 0:
            logger.warning(f"Invalid packet size: {size}")
            return b''

        try:
            data = bytearray()
            remaining = size
            CHUNK_SIZE = 8192  # 8KB chunks

            while remaining > 0:
                chunk_size = min(remaining, CHUNK_SIZE)
                chunk = demo_file.read(chunk_size)
                if not chunk:
                    logger.warning(f"Incomplete packet data: expected {size}, got {size - remaining}")
                    return b''  # Return an empty bytes object instead of None
                data.extend(chunk)
                remaining -= len(chunk)

                # Log progress for large packets
                if size > 1024 * 1024:  # 1MB
                    progress = ((size - remaining) / size) * 100
                    logger.debug(f"Reading large packet: {progress:.1f}% complete")

            return bytes(data)
        except Exception as e:
            logger.error(f"Error reading packet data: {e}", exc_info=True)
            return b''  # Return an empty bytes object instead of None
    
    def _attempt_packet_recovery(self, parser: 'DemoParser', demo_path: str) -> Optional[DemoPacket]:
        """Attempt to recover from corrupt packet by finding next valid packet"""
        MAX_RECOVERY_BYTES = 1024 * 1024  # 1MB max recovery scan
        bytes_scanned = 0
        buffer = bytearray()

        with open(demo_path, 'rb') as demo_file:
            while bytes_scanned < MAX_RECOVERY_BYTES:
                # Read one byte at a time looking for valid packet header
                byte = demo_file.read(1)
                if not byte:
                    return None  # EOF
                bytes_scanned += 1
                buffer.extend(byte)

                if len(buffer) >= 9:  # Minimum packet header size
                    # Try to parse as packet header
                    header = parser._read_packet_header(demo_file)
                    if header:
                        cmd_type, tick, size = header
                        if parser._is_valid_packet_header(cmd_type, tick, size):
                            # Found potential valid packet header
                            logger.info(f"Recovered packet header at offset {bytes_scanned}")
                            # Reset file position to start of packet
                            demo_file.seek(-9, 1)
                            return DemoPacket(cmd_type=cmd_type, tick=tick, data=parser._read_packet_data(demo_file, size))

                # Remove oldest byte if buffer is full
                buffer = buffer[1:]

        logger.error("Failed to recover valid packet")
        return None
            
    def _parse_packet_header(self, header_data: bytes) -> Optional[Tuple[int, int, int]]:
        """Parse packet header with validation"""
        try:
            cmd_type = header_data[0]  # First byte is command type
            tick = struct.unpack('<I', header_data[1:5])[0]  # Next 4 bytes are tick number
            size = struct.unpack('<I', header_data[5:9])[0]  # Last 4 bytes are size
        
            # Basic sanity checks
            if cmd_type > 32:  # Assuming 32 is the max valid command type
                raise ValueError(f"Invalid command type: {cmd_type}")
            if tick < 0:
                raise ValueError(f"Invalid tick number: {tick}")
            
            return cmd_type, tick, size
        except Exception as e:
            logger.error(f"Error parsing packet header: {e}")
            return None
        
    def _is_valid_packet_size(self, size: int) -> bool:
        """Validate packet size"""
        MIN_PACKET_SIZE = 0
        MAX_PACKET_SIZE = 1024 * 1024 * 10  # 10MB max packet size
    
        return MIN_PACKET_SIZE <= size <= MAX_PACKET_SIZE

    
    def _read_packet_header(self, demo_file: BinaryIO) -> Optional[Tuple[int, int, int]]:
        """Parse packet header with validation"""
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

            # Validate packet header
            if not self._is_valid_packet_header(cmd_type, tick, size):
                logger.warning(f"Invalid packet header: type={cmd_type}, tick={tick}, size={size}")
                return None

            return cmd_type, tick, size
        except Exception as e:
            logger.error(f"Error reading packet header: {e}", exc_info=True)
            return None
        
    def _is_valid_packet_header(self, cmd_type: int, tick: int, size: int) -> bool:
        """Validate packet header values"""
        # Define valid packet types
        VALID_TYPES = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32}

        # Type validation
        if cmd_type not in VALID_TYPES:
            return False

        # Tick validation - some packet types can have special tick values
        if cmd_type not in {3, 4, 9}:  # Normal packets
            if tick < 0 or tick > 1_000_000:  # Reasonable tick range
                return False

        # Size validation - different limits for different types
        if size <= 0:
            return False

        if cmd_type in {11, 12}:  # Table packets
            MAX_SIZE = 5 * 1024 * 1024  # 5MB
        elif cmd_type == 32:  # Custom data
            MAX_SIZE = 10 * 1024 * 1024  # 10MB
        else:  # Regular packets
            MAX_SIZE = 1024 * 1024  # 1MB

        if size > MAX_SIZE:
            return False

        return True


    def _process_packet(self, packet: DemoPacket) -> None:
        try:
            if packet.cmd_type == 7:  # Command packet
                self._handle_command_packet(packet.data)
            else:  # Data packet
                self._handle_data_packet(packet.data)
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
            self._translate_packet_data(packet)
            self._attempt_packet_recovery(self, self.demo_path)

    def _translate_packet_data(self, packet: DemoPacket) -> None:
        """Translate the raw packet data into human-readable format"""
        try:
            cmd_type = DemoPacketType.get_name(packet.cmd_type)
            logger.info(f"Packet Type: {cmd_type}")
            logger.info(f"Tick: {packet.tick}")

            # Translate the packet data
            data_bytes = packet.data
            data_str = ' '.join(f'{b:02X}' for b in data_bytes)
            logger.info(f"Packet Data (hex): {data_str}")

            # Try to parse the data based on the packet type
            if cmd_type == 'DEM_PACKET':
                self._translate_game_events(data_bytes, packet.tick)
            elif cmd_type == 'DEM_STRINGTABLES':
                self._translate_string_tables(data_bytes)
            elif cmd_type == 'DEM_DATATABLES':
                self._translate_data_tables(data_bytes)
            else:
                logger.info(f"Packet data (ASCII): {data_bytes.decode('ascii', errors='replace')}")

        except Exception as e:
            logger.error(f"Error translating packet data: {e}", exc_info=True)
        
    class CustomDataHandler(DemoMessageTypeHandler):
        def handle_packet(self, packet: DemoPacket) -> None:
            """Handle custom data packets (type 32)"""
            try:
                # Add your custom data handling logic here
                # For now, just log the data size
                logger.debug(f"Received custom data packet: {len(packet.data)} bytes")
            except Exception as e:
                logger.error(f"Error handling custom data packet: {e}")
    
    def _read_cmd_info(self, demo_file: BinaryIO) -> Optional[Tuple[int, int, int]]:
        """Read command info from demo file"""
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
        
    def _handle_data_packet(self, data: bytes) -> None:
        """Handle PBDEMS data packet"""
        if len(data) < 4:
            return
            
        try:
            packet_type = data[0]
            logger.debug(f"Processing data packet type {packet_type}")
            # Add data packet handling here
            pass
        except Exception as e:
            logger.error(f"Error handling data packet: {e}")
        
    def _create_packet(self, cmd_type: int, tick: int, data: bytes) -> bytes:
        """Create a packet with header and data"""
        header = struct.pack('<BII', cmd_type, tick, len(data))
        return header + data
    
    def _decode_packet(self, packet: bytes) -> List[Dict[str, Any]]:
        """
        Decode a demo packet into game events
        
        Args:
            packet: Raw packet bytes
            
        Returns:
            List[Dict[str, Any]]: List of parsed game events
        """
        events = []
        try:
            if len(packet) < 9:
                logger.warning("Packet too short for header")
                return events
            
            cmd_type, tick, size = struct.unpack('<BII', packet[:9])
            data = packet[9:]
            
            if len(data) < size:
                logger.warning(f"Incomplete packet data: expected {size} bytes, got {len(data)}")
                return events
            
            # Handle different message types
            if cmd_type == DemoMessageType.DEM_PACKET.value:
                events.extend(self._parse_game_events(data, tick))
            elif cmd_type == DemoMessageType.DEM_STRINGTABLES.value:
                self._update_string_tables(data)
            elif cmd_type == DemoMessageType.DEM_DATATABLES.value:
                self._update_data_tables(data)
            
        except Exception as e:
            logger.error(f"Error decoding packet: {e}", exc_info=True)
        
        return events
    
    def _parse_game_events(self, data: bytes, tick: int) -> List[Dict[str, Any]]:
        """
        Parse game events from packet data
        
        Args:
            data: Raw event data
            tick: Current tick number
            
        Returns:
            List[Dict[str, Any]]: List of parsed events
        """
        events = []
        offset = 0
        
        try:
            while offset + 2 < len(data):
                # Read event header
                event_type = int.from_bytes(data[offset:offset+1], byteorder='little')
                event_size = int.from_bytes(data[offset+1:offset+2], byteorder='little')
                
                # Validate event size
                if event_size <= 0 or offset + 2 + event_size > len(data):
                    logger.warning(f"Invalid event size {event_size} at offset {offset}")
                    break
                
                # Extract event data
                event_data = data[offset+2:offset+2+event_size]
                
                # Parse event
                try:
                    event_type_enum = EventType.from_id(event_type)
                    parsed_data = self._parse_event_data(event_type_enum, event_data)
                    
                    if parsed_data is not None:
                        event = {
                            'tick': tick,
                            'type': event_type_enum.name,
                            'data': parsed_data
                        }
                        
                        # Add position if event type requires it
                        if event_type_enum.requires_position:
                            position = self._read_vector(event_data)
                            if position:
                                event['position'] = position.to_dict()
                        
                        events.append(event)
                    
                except Exception as e:
                    logger.error(f"Error parsing event type {event_type}: {e}", exc_info=True)
                
                offset += 2 + event_size
                
        except Exception as e:
            logger.error(f"Error parsing game events: {e}", exc_info=True)
        
        return events

    def _read_vector(self, data: bytes, offset: int = 0) -> Optional[Vector3]:
        """
        Read a 3D vector from byte data
        
        Args:
            data: Raw bytes containing vector data
            offset: Starting position in bytes
            
        Returns:
            Optional[Vector3]: Parsed vector or None if invalid
        """
        try:
            if len(data) < offset + 12:  # Need at least 12 bytes for 3 floats
                logger.warning("Insufficient data for vector")
                return None

            x = struct.unpack('<f', data[offset:offset+4])[0]
            y = struct.unpack('<f', data[offset+4:offset+8])[0]
            z = struct.unpack('<f', data[offset+8:offset+12])[0]

            # Basic validation for reasonable values
            if any(abs(v) > 16384 for v in (x, y, z)):  # Max map size in units
                logger.warning(f"Vector values out of reasonable range: ({x}, {y}, {z})")
                return None

            return Vector3(x=x, y=y, z=z)

        except struct.error as e:
            logger.error(f"Error unpacking vector data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading vector: {e}")
            return None
    
    def _parse_event_data(self, event_type: EventType, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse event data based on event type
    
        Args:
            event_type: Type of event to parse
            data: Raw event data bytes
        
        Returns:
            Optional[Dict[str, Any]]: Parsed event data or None if parsing failed
        """
        try:
            # Map event types to their specific parsers
            event_parsers = {
                EventType.ROUND_START: self._parse_round_start_event,
                EventType.ROUND_END: self._parse_round_end_event,
                EventType.BOMB_PLANTED: self._parse_bomb_planted_event,
                EventType.BOMB_DEFUSED: self._parse_bomb_defused_event,
                EventType.PLAYER_DEATH: self._parse_player_death_event,
                EventType.PLAYER_POSITION: self._parse_player_position_event,
                EventType.SMOKE_START: self._parse_smoke_event,
                EventType.FLASH_EXPLODE: self._parse_flash_event,
                EventType.MOLOTOV_DETONATE: self._parse_molotov_event,
                EventType.GAME_START: self._parse_game_phase_change_event
            }
        
            # Get appropriate parser
            parser = event_parsers.get(event_type)
            if parser:
                return parser(data)
            else:
                logger.debug(f"No specific parser for event type {event_type.name}")
                return {}

        except Exception as e:
            logger.error(f"Error parsing event data for {event_type.name}: {e}", exc_info=True)
            return None

    def _parse_round_end_event(self, data: bytes) -> Dict[str, Any]:
        """Parse round end event data"""
        try:
            return {
                'winner': int.from_bytes(data[0:4], byteorder='little'),
                'reason': int.from_bytes(data[4:8], byteorder='little'),
                'message': self._read_string(data[8:], 10)  # Updated to call _read_string directly on bytes
            }
        except Exception as e:
            logger.error(f"Error parsing round end event: {e}", exc_info=True)
            return {}


    def _parse_smoke_event(self, data: bytes) -> Dict[str, Any]:
        """Parse smoke grenade event data"""
        position = self._read_vector(data, 4)
        return {
            'player_id': int.from_bytes(data[0:4], byteorder='little'),
            'position': position.to_dict() if position else None,
            'duration': struct.unpack('<f', data[16:20])[0] if len(data) >= 20 else 0
        }

    def _parse_bomb_defused(self, data: bytes) -> Dict[str, Any]:
        """
        Parse bomb defused event data
    
        Args:
            data: Raw event data bytes
        
        Returns:
            Dict[str, Any]: Parsed bomb defuse event data containing:
            - player_id: ID of the player who defused
            - site: Bomb site identifier ('A' or 'B')
            - x, y, z: Position coordinates of the defuse
            - time_remaining: Time that was left on bomb timer
        """
        try:
            return {
                'player_id': int.from_bytes(data[0:4], byteorder='little'),
                'site': chr(data[4]),  # 'A' or 'B'
                'x': struct.unpack('<f', data[5:9])[0],
                'y': struct.unpack('<f', data[9:13])[0],
                'z': struct.unpack('<f', data[13:17])[0],
                'time_remaining': struct.unpack('<f', data[17:21])[0] if len(data) >= 21 else None
            }
        except Exception as e:
            logger.error(f"Error parsing bomb defused event: {e}", exc_info=True)
            # Return minimal data if parsing fails
            return {
                'player_id': int.from_bytes(data[0:4], byteorder='little'),
                'site': '?'
            }
        
    def _parse_player_position(self, data: bytes) -> Dict[str, Any]:
        """Parse player position event data"""
        return {
            'player_id': int.from_bytes(data[0:4], byteorder='little'),
            'x': struct.unpack('<f', data[4:8])[0],
            'y': struct.unpack('<f', data[8:12])[0],
            'z': struct.unpack('<f', data[12:16])[0],
            'angle_x': struct.unpack('<f', data[16:20])[0],
            'angle_y': struct.unpack('<f', data[20:24])[0]
        }
    
    def _parse_player_death(self, data: bytes) -> Dict[str, Any]:
        """Parse player death event data"""
        return {
            'victim_id': int.from_bytes(data[0:4], byteorder='little'),
            'killer_id': int.from_bytes(data[4:8], byteorder='little'),
            'weapon': self._read_string(data[8:]),
            'headshot': bool(data[-1])
        }
    
    def _parse_bomb_planted(self, data: bytes) -> Dict[str, Any]:
        """Parse bomb planted event data"""
        return {
            'player_id': int.from_bytes(data[0:4], byteorder='little'),
            'site': chr(data[4]),  # 'A' or 'B'
            'x': struct.unpack('<f', data[5:9])[0],
            'y': struct.unpack('<f', data[9:13])[0],
            'z': struct.unpack('<f', data[13:17])[0]
        }
    
    def _parse_utility_event(self, data: bytes) -> Dict[str, Any]:
        """Parse utility event data"""
        return {
            'player_id': int.from_bytes(data[0:4], byteorder='little'),
            'x': struct.unpack('<f', data[4:8])[0],
            'y': struct.unpack('<f', data[8:12])[0],
            'z': struct.unpack('<f', data[12:16])[0]
        }
    
    def _parse_molotov_event(self, data: bytes) -> Dict[str, Any]:
        """Parse molotov event data"""
        return {
            'player_id': int.from_bytes(data[0:4], byteorder='little'),
            'x': struct.unpack('<f', data[4:8])[0],
            'y': struct.unpack('<f', data[8:12])[0],
            'z': struct.unpack('<f', data[12:16])[0],
            'duration': struct.unpack('<f', data[16:20])[0]
        }
    
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

    def _parse_round_start_event(self, data: bytes) -> Dict[str, Any]:
        """Parse round start event data"""
        try:
            objective_byte = data[8:9]  # Get a single byte as bytes, not int
            return {
                'timelimit': int.from_bytes(data[0:4], byteorder='little'),
                'fraglimit': int.from_bytes(data[4:8], byteorder='little'),
                'objective': objective_byte.decode('ascii')
            }
        except Exception as e:
            logger.error(f"Error parsing round start event: {e}")
            return {
                'timelimit': 0,
                'fraglimit': 0,
                'objective': ''
            }

    def _parse_bomb_planted_event(self, data: bytes) -> Dict[str, Any]:
        """Parse bomb planted event data"""
        return {
            'player_id': int.from_bytes(data[0:4], byteorder='little'),
            'site': chr(data[4]),
            'x': struct.unpack('<f', data[5:9])[0],
            'y': struct.unpack('<f', data[9:13])[0],
            'z': struct.unpack('<f', data[13:17])[0]
        }
    def _parse_bomb_defused_event(self, data: bytes) -> Dict[str, Any]:
        """Parse bomb defused event data"""
        return {
            'player_id': int.from_bytes(data[0:4], byteorder='little'),
            'site': chr(data[4]),
            'x': struct.unpack('<f', data[5:9])[0],
            'y': struct.unpack('<f', data[9:13])[0],
            'z': struct.unpack('<f', data[13:17])[0]
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
    def _parse_flash_event(self, data: bytes) -> Dict[str, Any]:
        """Parse flashbang event data"""
        return {
            'player_id': int.from_bytes(data[0:4], byteorder='little'),
            'x': struct.unpack('<f', data[4:8])[0],
            'y': struct.unpack('<f', data[8:12])[0],
            'z': struct.unpack('<f', data[12:16])[0],
            'duration': struct.unpack('<f', data[16:20])[0]
        }

    def _read_string(self, data: bytes, length: Optional[int] = None) -> str:
        """Read a fixed-length string from byte data, or until end if length is None."""
        if length is None:
            length = len(data)
        buffer = data[:length]  # Slice to the specified length
        return buffer.decode('ascii', errors='replace').strip('\0')
    
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
    def _parse(self) -> Dict[str, Any]:
        """Parse the demo file"""
        try:
            with open(self.demo_path, 'rb') as demo_file:
                # Parse the demo header
                self.header = self._parse_demo_header(demo_file)
                if not self.header:
                    raise ValueError("Failed to parse demo header")

                # Process packets
                packets_processed = 0
                errors_encountered = 0
                while True:
                    packet = self._read_packet(demo_file)
                    if not packet:
                        if errors_encountered > 100:
                            logger.error("Too many errors, stopping parse")
                            break
                        else:
                            errors_encountered += 1
                            continue
                    self._process_packet(packet)
                    packets_processed += 1

                # Perform analysis
                analysis = {
                    'total_packets_processed': packets_processed,
                    'total_errors_encountered': errors_encountered,
                    'final_position': demo_file.tell(),
                    'file_size': os.path.getsize(self.demo_path)
                }

                return {
                    'header': self.header.to_dict(),
                    'statistics': analysis
                }

        except Exception as e:
            logger.error(f"Error parsing demo: {e}", exc_info=True)
            return {}
        
    def _read_line(self, demo_file: BinaryIO) -> Optional[bytes]:
        """Read a line of text from the demo file"""
        line = b''
        while True:
            char = demo_file.read(1)
            if not char or char == b'\n':
                return line
            line += char
    
        
    def _is_valid_ip_port(self, address: str) -> bool:
        """Check if the given address is a valid IP:Port format"""
        return bool(re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}', address))
            
def main():
    if len(sys.argv) > 1:
        demo_path = sys.argv[1]
    else:
        # Look for .dem files in current directory
        dem_files = [f for f in os.listdir('.') if f.endswith('.dem')]
        if dem_files:
            demo_path = dem_files[0]
            logger.info(f"Found demo file: {demo_path}")
        else:
            logger.error("No demo file provided and no .dem files found in current directory")
            sys.exit(1)

    try:
        parser = DemoParser(demo_path)
        analysis = parser._parse()

        # Print results
        print("\nDemo Analysis Results:")
        print("----------------------")
        print(f"Map: {parser.header.map_name if parser.header else 'Unknown'}")
        print(f"Total Events: {len(parser.events)}")
        print(f"Total Rounds: {len(parser.rounds)}")

        # Print some event statistics
        event_types = {}
        for event in parser.events:
            event_type = event.get('type', 'Unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1

        print("\nEvent Breakdown:")
        for event_type, count in event_types.items():
            print(f"{event_type}: {count}")

    except FileNotFoundError:
        logger.error(f"Demo file not found: {demo_path}")
    except DemoParserException as e:
        logger.error(f"Error parsing demo: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    main()