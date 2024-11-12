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
    """Base exception for demo parsing errors"""
    pass

class DemoParserCorruptedFileException(DemoParserException):
    """Exception for corrupted demo files"""
    pass

class PacketReadError(Exception):
    """Exception for packet reading errors"""
    pass

@dataclass
class DemoPacket:
    """Represents a parsed demo packet"""
    cmd_type: int
    tick: int
    data: bytes
    
    @classmethod
    def from_bytes(cls, data: bytes, format_type: str) -> Optional['DemoPacket']:
        """Create packet from bytes with format-specific parsing"""
        try:
            if format_type == "PBDEMS2":
                # PBDEMS2 packet format
                if len(data) < 9:  # Minimum size for header
                    raise PacketReadError("Packet too small for header")
                    
                cmd_type = data[0]
                tick = int.from_bytes(data[1:5], byteorder='little')
                size = int.from_bytes(data[5:9], byteorder='little')
                
                if size <= 0 or size > len(data) - 9:
                    raise PacketReadError(f"Invalid packet size: {size}")
                    
                packet_data = data[9:9+size]
                
            else:  # HL2DEMO format
                if len(data) < 6:
                    raise PacketReadError("Packet too small for header")
                    
                cmd_type = data[0]
                tick = int.from_bytes(data[1:5], byteorder='little')
                packet_data = data[5:]
                
            return cls(
                cmd_type=cmd_type,
                tick=tick,
                data=packet_data
            )
            
        except Exception as e:
            logger.error(f"Error parsing packet: {e}")
            raise PacketReadError(f"Failed to parse packet: {e}")

class DemoMessageTypeHandler:
    """Base class for packet type handlers"""
    def __init__(self, parser: 'DemoParser'):
        self.parser = parser
        self.format_type = parser.header.format.value if parser.header and parser.header.format else "HL2DEMO"

    def handle_packet(self, packet: DemoPacket) -> None:
        """Handle a demo packet"""
        raise NotImplementedError()
        
    def _validate_packet(self, packet: DemoPacket) -> bool:
        """Validate packet data before handling"""
        if not packet.data:
            logger.warning("Empty packet data")
            return False
            
        if self.format_type == "PBDEMS2":
            # Add any PBDEMS2-specific validation
            return len(packet.data) >= 4  # Minimum size for any meaningful data
        else:
            # HL2DEMO validation
            return len(packet.data) >= 1  # Minimum size for HL2DEMO
            
class GameEventHandler(DemoMessageTypeHandler):
    """Handler for game events"""
    def handle_packet(self, packet: DemoPacket) -> None:
        if not self._validate_packet(packet):
            return
            
        try:
            events = self.parser._decode_packet(packet.data)
            for event in events:
                if event:  # Skip None or empty events
                    self.parser._handle_event(event)
        except Exception as e:
            logger.error(f"Error handling game event packet: {e}")
            raise DemoParserException(f"Game event parsing failed: {e}")

class StringTableHandler(DemoMessageTypeHandler):
    """Handler for string table updates"""
    def handle_packet(self, packet: DemoPacket) -> None:
        if not self._validate_packet(packet):
            return
            
        try:
            if self.format_type == "PBDEMS2":
                # PBDEMS2 string table format
                self._handle_pbdems2_string_table(packet.data)
            else:
                # Original HL2DEMO format
                self.parser._update_string_tables(packet.data)
        except Exception as e:
            logger.error(f"Error handling string table packet: {e}")
            raise DemoParserException(f"String table parsing failed: {e}")
            
    def _handle_pbdems2_string_table(self, data: bytes) -> None:
        """Handle PBDEMS2 format string tables"""
        if len(data) < 4:
            logger.warning("String table data too small")
            return
            
        try:
            num_tables = int.from_bytes(data[:4], byteorder='little')
            logger.debug(f"Processing {num_tables} string tables")
            offset = 4
            
            for table_index in range(num_tables):
                if offset + 8 > len(data):
                    logger.warning(f"Insufficient data for string table {table_index}")
                    break
                    
                table_id = int.from_bytes(data[offset:offset+4], byteorder='little')
                num_entries = int.from_bytes(data[offset+4:offset+8], byteorder='little')
                offset += 8
                
                logger.debug(f"String table {table_index}: ID={table_id}, Entries={num_entries}")
                
                # Extract the portion of data for this table
                table_data = data[offset:]
                # Call the original method with just the data
                self.parser._update_string_tables(table_data)
                
        except struct.error as e:
            logger.error(f"Error parsing string table structure: {e}")
            raise DemoParserException(f"String table structure error: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid value in string table: {e}")
            raise DemoParserException(f"String table value error: {e}") from e
        except AttributeError as e:
            logger.error(f"Parser method error: {e}")
            raise DemoParserException(f"String table parsing error: {e}") from e
                
        except Exception as e:
            logger.error(f"Error parsing PBDEMS2 string table: {e}")
            raise DemoParserException(f"PBDEMS2 string table parsing failed: {e}")

class DataTableHandler(DemoMessageTypeHandler):
    """Handler for data table updates"""
    def handle_packet(self, packet: DemoPacket) -> None:
        if not self._validate_packet(packet):
            return
            
        try:
            if self.format_type == "PBDEMS2":
                # PBDEMS2 data table format
                self._handle_pbdems2_data_table(packet.data)
            else:
                # Original HL2DEMO format
                self.parser._update_data_tables(packet.data)
        except Exception as e:
            logger.error(f"Error handling data table packet: {e}")
            raise DemoParserException(f"Data table parsing failed: {e}")
            
    def _handle_pbdems2_data_table(self, data: bytes) -> None:
        """Handle PBDEMS2 format data tables"""
        if len(data) < 4:
            logger.warning("Data table packet too small")
            return
            
        try:
            num_tables = int.from_bytes(data[:4], byteorder='little')
            logger.debug(f"Processing {num_tables} data tables")
            offset = 4
            
            for table_index in range(num_tables):
                if offset + 8 > len(data):
                    logger.warning(f"Insufficient data for data table {table_index}")
                    break
                    
                table_id = int.from_bytes(data[offset:offset+4], byteorder='little')
                num_entries = int.from_bytes(data[offset+4:offset+8], byteorder='little')
                offset += 8
                
                logger.debug(f"Data table {table_index}: ID={table_id}, Entries={num_entries}")
                
                # Extract the portion of data for this table
                table_data = data[offset:]
                # Call the original method with just the data
                self.parser._update_data_tables(table_data)
                
        except struct.error as e:
            logger.error(f"Error parsing data table structure: {e}")
            raise DemoParserException(f"Data table structure error: {e}") from e
        except ValueError as e:
            logger.error(f"Invalid value in data table: {e}")
            raise DemoParserException(f"Data table value error: {e}") from e
        except AttributeError as e:
            logger.error(f"Parser method error: {e}")
            raise DemoParserException(f"Data table parsing error: {e}") from e
                
        except Exception as e:
            logger.error(f"Error parsing PBDEMS2 data table: {e}")
            raise DemoParserException(f"PBDEMS2 data table parsing failed: {e}")


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

class DemoFormat(Enum):
    HL2DEMO = "HL2DEMO"
    PBDEMS2 = "PBDEMS2"

@dataclass
class DemoHeader:
    """
    CS2 Demo file header following official format specification.
    References:
    - HL2DEMO format: https://developer.valvesoftware.com/wiki/DEM_Format
    - String length is 260 chars for compatibility fields
    """
    
    # Format specification from wiki
    MAGIC_SIZE: ClassVar[int] = 8           # "HL2DEMO" or "PBDEMS2" + null terminator
    STRING_LENGTH: ClassVar[int] = 260      # Fixed string length for backward compatibility
    HEADER_SIZE: ClassVar[int] = 1072       # Total header size (documented in wiki)
    SUPPORTED_FORMATS = ["HL2DEMO", "PBDEMS2"]
    
    # Offset constants based on wiki specification
    MAGIC_OFFSET: ClassVar[int] = 0
    DEMO_PROTOCOL_OFFSET: ClassVar[int] = 8
    NETWORK_PROTOCOL_OFFSET: ClassVar[int] = 12
    SERVER_NAME_OFFSET: ClassVar[int] = 16
    CLIENT_NAME_OFFSET: ClassVar[int] = SERVER_NAME_OFFSET + STRING_LENGTH
    MAP_NAME_OFFSET: ClassVar[int] = CLIENT_NAME_OFFSET + STRING_LENGTH
    GAME_DIR_OFFSET: ClassVar[int] = MAP_NAME_OFFSET + STRING_LENGTH
    PLAYBACK_TIME_OFFSET: ClassVar[int] = GAME_DIR_OFFSET + STRING_LENGTH
    TICKS_OFFSET: ClassVar[int] = PLAYBACK_TIME_OFFSET + 4
    FRAMES_OFFSET: ClassVar[int] = TICKS_OFFSET + 4
    SIGNON_LENGTH_OFFSET: ClassVar[int] = FRAMES_OFFSET + 4
    
    # Fields with default values
    raw_data: bytes = field(repr=False)
    format: Optional[DemoFormat] = field(default=None)
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
        """Validate and parse header data after initialization"""
        # Log raw header data for debugging
        logger.debug(f"Raw header size: {len(self.raw_data)} bytes")
        logger.debug(f"First 16 bytes: {self.raw_data[:16].hex()}")
        
        # Parse magic field
        magic_bytes = self.raw_data[self.MAGIC_OFFSET:self.MAGIC_OFFSET + self.MAGIC_SIZE]
        logger.debug(f"Magic bytes: {magic_bytes.hex()}")
        
        # For PBDEMS2 specifically, preserve spaces in magic string
        self.magic = magic_bytes.decode('ascii', errors='replace')
        logger.info(f"Decoded magic string: '{self.magic}'")
        
        # Detect format
        self.format = self._detect_format(self.magic.upper())
        if self.format is None:
            # On failure, try alternate parsing methods
            alt_magic = magic_bytes.decode('ascii', errors='replace').strip('\0').strip()
            logger.info(f"Trying alternate magic parsing: '{alt_magic}'")
            self.format = self._detect_format(alt_magic.upper())
            
            if self.format is None:
                raise ValueError(
                    f"Invalid demo file magic: '{self.magic}' "
                    f"(hex: {magic_bytes.hex()})"
                )
        
        # Parse common fields
        try:
            self.demo_protocol = int.from_bytes(
                self.raw_data[self.DEMO_PROTOCOL_OFFSET:self.NETWORK_PROTOCOL_OFFSET],
                byteorder='little'
            )
            self.network_protocol = int.from_bytes(
                self.raw_data[self.NETWORK_PROTOCOL_OFFSET:self.SERVER_NAME_OFFSET],
                byteorder='little'
            )
            logger.info(f"Protocols - Demo: {self.demo_protocol}, Network: {self.network_protocol}")
        except Exception as e:
            logger.error(f"Error parsing protocol versions: {e}")
            raise
        
        # Format-specific parsing
        if self.format == DemoFormat.PBDEMS2:
            self._parse_pbdems2_format()
        else:
            self._parse_hl2demo_format()
            
        # Parse common fields regardless of format
        self.demo_protocol = struct.unpack('i', self.raw_data[self.DEMO_PROTOCOL_OFFSET:self.NETWORK_PROTOCOL_OFFSET])[0]
        self.network_protocol = struct.unpack('i', self.raw_data[self.NETWORK_PROTOCOL_OFFSET:self.SERVER_NAME_OFFSET])[0]
        
        if self.format == DemoFormat.HL2DEMO:
            self._parse_hl2demo_format()
        else:
            self._parse_pbdems2_format()

    def _detect_format(self, magic_upper: str) -> Optional[DemoFormat]:
        """Detect the demo format from magic string with detailed logging"""
        # Log raw input
        logger.info(f"Raw magic string: '{magic_upper}'")
        
        # For PBDEMS2, we accept the exact string "PBDEMS2    " (with trailing spaces)
        if magic_upper == "PBDEMS2    ":
            logger.info("Detected exact PBDEMS2 format with spaces")
            return DemoFormat.PBDEMS2
            
        # Clean up magic string for other cases
        magic_clean = magic_upper.strip('\0').strip()
        logger.info(f"Cleaned magic string: '{magic_clean}'")
        
        if magic_clean == "HL2DEMO":
            logger.info("Detected HL2DEMO format")
            return DemoFormat.HL2DEMO
        elif magic_clean == "PBDEMS2":
            logger.info("Detected PBDEMS2 format (cleaned)")
            return DemoFormat.PBDEMS2
            
        logger.warning(f"Unknown format. Raw: '{magic_upper}', Cleaned: '{magic_clean}'")
        return None

    def _parse_hl2demo_format(self) -> None:
        """Parse header fields for HL2DEMO format"""
        # All fields have fixed positions in HL2DEMO format
        self.server_name = self.raw_data[self.SERVER_NAME_OFFSET:self.CLIENT_NAME_OFFSET].decode('ascii', errors='replace').strip('\0')
        self.client_name = self.raw_data[self.CLIENT_NAME_OFFSET:self.MAP_NAME_OFFSET].decode('ascii', errors='replace').strip('\0')
        self.map_name = self.raw_data[self.MAP_NAME_OFFSET:self.GAME_DIR_OFFSET].decode('ascii', errors='replace').strip('\0')
        self.game_directory = self.raw_data[self.GAME_DIR_OFFSET:self.PLAYBACK_TIME_OFFSET].decode('ascii', errors='replace').strip('\0')
        self.playback_time = struct.unpack('f', self.raw_data[self.PLAYBACK_TIME_OFFSET:self.TICKS_OFFSET])[0]
        self.ticks = struct.unpack('i', self.raw_data[self.TICKS_OFFSET:self.FRAMES_OFFSET])[0]
        self.frames = struct.unpack('i', self.raw_data[self.FRAMES_OFFSET:self.SIGNON_LENGTH_OFFSET])[0]
        self.signon_length = struct.unpack('i', self.raw_data[self.SIGNON_LENGTH_OFFSET:self.SIGNON_LENGTH_OFFSET + 4])[0]

    def _parse_pbdems2_format(self) -> None:
        """Parse header fields for PBDEMS2 format"""
        # Search for map name with de_ prefix
        map_data = self.raw_data[self.SERVER_NAME_OFFSET:]
        map_start = map_data.find(b'de_')
        if map_start >= 0:
            map_end = map_data.find(b'\0', map_start)
            if map_end >= 0:
                self.map_name = map_data[map_start:map_end].decode('ascii', errors='replace')

        # Look for server name indicators
        for prefix in [b'FACEIT', b'Valve', b'ESL', b'ESEA']:
            server_start = map_data.find(prefix)
            if server_start >= 0:
                server_end = map_data.find(b'\0', server_start)
                if server_end >= 0:
                    self.server_name = map_data[server_start:server_end].decode('ascii', errors='replace')
                    break

        # Game directory is typically after a path prefix
        for prefix in [b'/game/', b'/csgo/', b'/cs2/']:
            dir_start = map_data.find(prefix)
            if dir_start >= 0:
                dir_end = map_data.find(b'\0', dir_start)
                if dir_end >= 0:
                    self.game_directory = map_data[dir_start:dir_end].decode('ascii', errors='replace')
                    break

    @classmethod
    def dump_header_bytes(cls, raw_data: bytes) -> str:
        """Dump header bytes in a readable format for debugging"""
        lines = []
        
        # Add summary
        lines.append(f"Header size: {len(raw_data)} bytes")
        
        # Hex dump with ASCII representation
        for i in range(0, len(raw_data), 16):
            chunk = raw_data[i:i+16]
            hex_values = ' '.join(f'{b:02x}' for b in chunk)
            ascii_values = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            lines.append(f"{i:04x}: {hex_values:48s} {ascii_values}")
            
            # Add extra spacing every 4 lines for readability
            if i % 64 == 48:
                lines.append("")
                
        return '\n'.join(lines)

    @classmethod
    def from_bytes(cls, raw_data: bytes) -> 'DemoHeader':
        """Create a DemoHeader instance from raw bytes with enhanced logging"""
        # Dump header bytes for debugging
        logger.debug("Header dump:\n" + cls.dump_header_bytes(raw_data[:256]))  # First 256 bytes
        
        try:
            return cls(raw_data=raw_data)
        except Exception as e:
            logger.error("Failed to parse header. Raw data:\n" + 
                      cls.dump_header_bytes(raw_data[:64]))  # Show first 64 bytes on error
            raise

    @classmethod
    def from_file(cls, demo_path: str | Path) -> 'DemoHeader':
        """Create header directly from a demo file"""
        path = Path(demo_path)
        if not path.exists():
            raise FileNotFoundError(f"Demo file not found: {path}")
            
        try:
            with path.open('rb') as f:
                header_data = f.read(cls.HEADER_SIZE)
                return cls.from_bytes(header_data)
        except Exception as e:
            raise ValueError(f"Failed to read demo file: {str(e)}") from e

    def to_dict(self) -> dict:
        """Convert header to a dictionary format for serialization"""
        return {
            'format': self.format.value if self.format else 'UNKNOWN',
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
        """Human-readable string representation"""
        format_str = self.format.value if self.format else 'UNKNOWN'
        return (
            f"CS2 Demo ({format_str})\n"
            f"Map: {self.map_name}\n"
            f"Server: {self.server_name}\n"
            f"Duration: {self.playback_time:.2f}s\n"
            f"Ticks: {self.ticks}"
        )
class DemoParser:
    """CS2 Demo Parser with comprehensive analysis capabilities"""

    HEADER_SIZE = 1072  # Updated correct header size
    TICK_RATE = 64
    PLAYERS_PER_TEAM = 5
    MAX_PACKET_SIZE = 1024 * 1024 
    
    def _read_data_table_entry(self, data: bytes, offset: int, table_id: int) -> Optional[DataTableEntry]:
        """
        Read a single data table entry from the given offset
        
        Args:
            data: Raw bytes containing entry data
            offset: Starting position in bytes
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
            entry_id = struct.unpack('<i', data[offset:offset+4])[0]
            type_id = struct.unpack('<i', data[offset+4:offset+8])[0]
            name_length = struct.unpack('<i', data[offset+8:offset+12])[0]

            # Validate name length
            if name_length <= 0 or offset + 12 + name_length > len(data):
                logger.warning(f"Invalid name length: {name_length}")
                return None

            # Read entry name
            name = data[offset+12:offset+12+name_length].decode('utf-8', errors='replace')
            current_offset = offset + 12 + name_length

            # Get data type
            try:
                data_type = DataType(type_id)
            except ValueError:
                logger.warning(f"Unknown data type ID: {type_id}")
                data_type = DataType.UNKNOWN

            # Parse data based on type
            parsed_data = self._parse_data_value(data[current_offset:], data_type)

            return DataTableEntry(
                entry_id=entry_id,
                name=name,
                data_type=data_type,
                data=parsed_data,
                table_id=table_id
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
                if len(data) < 4:
                    return 0
                return int.from_bytes(data[:4], byteorder='little')
                
            elif data_type == DataType.FLOAT:
                if len(data) < 4:
                    return 0.0
                return struct.unpack('<f', data[:4])[0]
                
            elif data_type == DataType.STRING:
                if len(data) < 4:
                    return ""
                str_length = int.from_bytes(data[:4], byteorder='little')
                if str_length <= 0 or str_length > len(data) - 4:
                    return ""
                return data[4:4+str_length].decode('utf-8', errors='replace')
                
            elif data_type == DataType.VECTOR:
                if len(data) < 12:
                    return Vector3(0, 0, 0)
                x = struct.unpack('<f', data[0:4])[0]
                y = struct.unpack('<f', data[4:8])[0]
                z = struct.unpack('<f', data[8:12])[0]
                return Vector3(x, y, z)
                
            elif data_type == DataType.BOOLEAN:
                if not data:
                    return False
                return bool(data[0])
                
            elif data_type == DataType.ARRAY:
                if len(data) < 8:
                    return []
                length = int.from_bytes(data[:4], byteorder='little')
                element_type_id = int.from_bytes(data[4:8], byteorder='little')
                try:
                    element_type = DataType(element_type_id)
                except ValueError:
                    logger.warning(f"Unknown array element type: {element_type_id}")
                    return []
                    
                array = []
                offset = 8
                for _ in range(length):
                    if offset >= len(data):
                        break
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

    def _get_string_table_value(self, table_id: int, entry_id: int) -> Optional[str]:
        """Get a value from the string tables"""
        try:
            return self.string_tables.get(table_id, {}).get(entry_id)
        except Exception as e:
            logger.warning(f"Error retrieving string table value: {e}")
            return None

    def _get_data_table_value(self, table_id: int, entry_id: int) -> Optional[Any]:
        """Get a value from the data tables"""
        try:
            return self.data_tables.get(table_id, {}).get(entry_id)
        except Exception as e:
            logger.warning(f"Error retrieving data table value: {e}")
            return None

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
        
        self.skip_corrupted = skip_corrupted
        self.header: Optional[DemoHeader] = None
        self.rounds: List[Round] = []
        self.current_round: int = 0
        self.events: List[GameEvent] = []
        self.current_tick: int = 0
        
        # Table storage
        self.string_tables: Dict[int, Dict[int, str]] = {}
        self.data_tables: Dict[int, Dict[int, Any]] = {}
        
        # Player state tracking
        self.players: Dict[int, PlayerState] = {}
        
        # Game state tracking
        self._retake_cache: Dict[int, bool] = {}
        self._position_cache: Dict[Tuple[float, float, float], bool] = {}
        
        # Message handlers
        self._message_type_handlers: Dict[int, DemoMessageTypeHandler] = {
            DemoPacketType.DEM_PACKET: GameEventHandler(self),
            DemoPacketType.DEM_STRINGTABLES: StringTableHandler(self),
            DemoPacketType.DEM_DATATABLES: DataTableHandler(self)
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
    
    def _debug_dump_bytes(self, data: bytes, offset: int = 0, length: int = 64) -> str:
        """Create a detailed hex dump of bytes for debugging"""
        result = []
        chunk_size = 16
        
        for i in range(0, min(length, len(data)), chunk_size):
            chunk = data[i:i+chunk_size]
            hex_values = ' '.join(f'{b:02x}' for b in chunk)
            ascii_values = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            result.append(f"{offset+i:04x}: {hex_values:48s} {ascii_values}")
            
        return '\n'.join(result)

    def _analyze_header_data(self, data: bytes) -> Dict[str, Any]:
        """Analyze demo header data in detail"""
        analysis = {}
        
        # Check magic
        magic_bytes = data[:8]
        analysis['magic_hex'] = ' '.join(f'{b:02x}' for b in magic_bytes)
        analysis['magic_ascii'] = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in magic_bytes)
        
        # Look for map name
        for i in range(len(data)-4):
            if data[i:i+3] in [b'de_', b'cs_']:
                # Found potential map name
                end = i
                while end < len(data) and data[end] != 0 and end < i + 64:
                    end += 1
                potential_map = data[i:end]
                analysis.setdefault('potential_maps', []).append({
                    'offset': i,
                    'bytes': potential_map.hex(),
                    'ascii': potential_map.decode('ascii', errors='replace')
                })
                
        # Log structure
        analysis['structure'] = self._debug_dump_bytes(data)
        
        return analysis

    def _find_string_in_data(self, data: bytes, prefixes: List[bytes], max_length: int = 64) -> Optional[str]:
        """Find and validate a string in binary data starting with given prefixes"""
        for prefix in prefixes:
            pos = data.find(prefix)
            if pos >= 0:
                # Found a prefix, now find the end
                end = pos
                for i in range(pos, min(pos + max_length, len(data))):
                    if data[i] == 0 or data[i] < 32:
                        end = i
                        break
                
                try:
                    string = data[pos:end].decode('ascii', errors='strict')
                    if all(c.isprintable() and c not in '<>:"/\\|?*' for c in string):
                        return string
                except UnicodeDecodeError:
                    continue
        return None
    
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
        """Parse demo header with detailed analysis"""
        try:
            # Read a larger chunk for analysis
            header_data = demo_file.read(2048)  # Read 2KB for analysis
            if len(header_data) < 16:
                logger.error("Not enough data for header")
                return None
                
            # Log initial analysis
            logger.debug("Header analysis:")
            logger.debug(self._debug_dump_bytes(header_data[:64]))
            
            # Try to detect format
            magic = header_data[:8].decode('ascii', errors='replace').rstrip('\0')
            logger.info(f"Magic string: '{magic}' (hex: {header_data[:8].hex()})")
            
            if 'PBDEMS' in magic:
                # PBDEMS2 format
                logger.info("Detected PBDEMS2 format")
                header = DemoHeader(
                    raw_data=header_data[:1072],  # Keep original header size
                    magic='PBDEMS2'
                )
                
                # Parse protocols
                header.demo_protocol = int.from_bytes(header_data[8:12], byteorder='little')
                header.network_protocol = int.from_bytes(header_data[12:16], byteorder='little')
                
                # Parse map name using strict rules
                header.map_name = self._parse_pbdems2_map_name(header_data[16:])
                
                # Find server name
                SERVER_PREFIXES = [b'FACEIT', b'Valve', b'ESL']
                for prefix in SERVER_PREFIXES:
                    pos = header_data.find(prefix)
                    if pos >= 0:
                        end = header_data.find(b'\0', pos)
                        if end > pos:
                            header.server_name = header_data[pos:end].decode('ascii', errors='replace')
                            break
                            
                logger.info(f"Parsed PBDEMS2 header - Map: {header.map_name}, Server: {header.server_name}")
                return header
                
            elif 'HL2DEMO' in magic:
                # HL2DEMO format - parse normally
                logger.info("Detected HL2DEMO format")
                if len(header_data) < 1072:
                    raise ValueError("Insufficient data for HL2DEMO header")
                    
                return DemoHeader(raw_data=header_data[:1072])
                
            else:
                logger.error(f"Unknown demo format: '{magic}'")
                return None
                
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

    def _ensure_player(self, player_id: int, player_name: Optional[str] = None, team: Optional[Team] = None) -> PlayerState:
        """Ensure a player exists in the state tracking, creating if necessary"""
        if player_id not in self.players:
            self.players[player_id] = PlayerState(
                player_id=player_id,
                name=player_name or f"Player_{player_id}",
                team=team or Team.UNKNOWN,
                position=Position(0, 0, 0),
                tick=self.current_tick
            )
        return self.players[player_id]
    
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
    
    def _update_player_position(self, event: GameEvent) -> None:
        """Update player position based on a PLAYER_POSITION event"""
        try:
            player_id = event.data['player_id']
            player_name = event.data.get('player_name', f"Player_{player_id}")
            team = Team.from_string(event.data.get('team', 'UNKNOWN'))
            
            # Get or create player state
            player = self._ensure_player(player_id, player_name, team)
            
            # Update position and tick
            player.position = event.position or player.position  # Keep existing position if new one is None
            player.tick = event.tick
            
            logger.debug(f"Updated position for player {player_id} ({player_name}): {player.position}")
            
        except KeyError as e:
            logger.warning(f"Missing required data in position event: {e}")
        except Exception as e:
            logger.error(f"Error updating player position: {e}")

    def _handle_player_death(self, event: GameEvent) -> None:
        """Handle a PLAYER_DEATH event"""
        try:
            killer_id = event.data.get('killer_id')
            victim_id = event.data.get('victim_id')
            
            if not victim_id:
                logger.warning("Death event missing victim ID")
                return
                
            # Update killer state
            if killer_id:
                killer = self._ensure_player(killer_id)
                killer.health = 100
                killer.armor = 100
                logger.debug(f"Reset health/armor for killer {killer_id}")
                
            # Update victim state
            victim = self._ensure_player(victim_id)
            victim.is_alive = False
            victim.health = 0
            logger.debug(f"Marked player {victim_id} as dead")
            
        except Exception as e:
            logger.error(f"Error handling player death: {e}")
    
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

    def _reset_player_states(self) -> None:
        """Reset player states for a new round"""
        for player in self.players.values():
            player.health = 100
            player.armor = 100
            player.is_alive = True
            player.has_defuser = False
            player.has_helmet = False
        logger.debug("Reset all player states for new round")

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
    
    def _is_valid_packet_header(self, cmd_type: int, tick: int, size: int) -> bool:
        """Validate packet header values with strict limits"""
        # Command type validation (PBDEMS2 specific types)
        VALID_CMD_TYPES = {
            2,   # DEM_STOP
            3,   # DEM_FILEHEADER
            4,   # DEM_FILEINFO
            5,   # DEM_SYNCTICK
            6,   # DEM_MESSAGE
            7,   # DEM_PACKET
            8,   # DEM_SIGNONPACKET
            9,   # DEM_CONSOLECMD
            10,  # DEM_USERCMD
            11,  # DEM_DATATABLES
            12,  # DEM_STRINGTABLES
            32   # PBDEMS_CUSTOMDATA
        }
        
        if cmd_type not in VALID_CMD_TYPES:
            logger.warning(f"Invalid command type: {cmd_type}")
            return False
            
        # Tick validation - CS2 uses reasonable tick values
        MAX_REASONABLE_TICK = 1_000_000  # ~4.3 hours at 64 tick
        if tick < 0 or tick > MAX_REASONABLE_TICK:
            logger.warning(f"Unreasonable tick value: {tick}")
            return False
            
        # Size validation - CS2 packets are reasonably sized
        MIN_PACKET_SIZE = 0
        MAX_PACKET_SIZE = 1024 * 1024 * 2  # 2MB max packet size
        if size < MIN_PACKET_SIZE or size > MAX_PACKET_SIZE:
            logger.warning(f"Invalid packet size: {size}")
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
        """Validate packet size with PBDEMS2 limits"""
        MIN_SIZE = 0
        MAX_SIZE = 1024 * 1024 * 2  # 2MB max packet size
        return MIN_SIZE <= size <= MAX_SIZE

    
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
            if len(data) < 4:
                logger.warning("String table data too small")
                return
                
            num_tables = int.from_bytes(data[offset:offset+4], byteorder='little')
            logger.debug(f"Processing {num_tables} string tables")
            offset += 4

            for _ in range(num_tables):
                if offset + 8 > len(data):
                    logger.warning("Insufficient data for string table")
                    break
                    
                table_id = int.from_bytes(data[offset:offset+4], byteorder='little')
                num_entries = int.from_bytes(data[offset+4:offset+8], byteorder='little')
                offset += 8
                
                logger.debug(f"String table {table_id}: {num_entries} entries")
                
                # Initialize table if needed
                if table_id not in self.string_tables:
                    self.string_tables[table_id] = {}
                    
                # Process entries
                for _ in range(num_entries):
                    if offset + 4 > len(data):
                        break
                        
                    entry_id = int.from_bytes(data[offset:offset+4], byteorder='little')
                    offset += 4
                    
                    # Read string length
                    if offset + 4 > len(data):
                        break
                    string_length = int.from_bytes(data[offset:offset+4], byteorder='little')
                    offset += 4
                    
                    # Read string
                    if offset + string_length > len(data):
                        break
                    try:
                        string_value = data[offset:offset+string_length].decode('utf-8', errors='replace')
                        offset += string_length
                        
                        # Store the string
                        self.string_tables[table_id][entry_id] = string_value
                        
                    except Exception as e:
                        logger.warning(f"Error decoding string in table {table_id}: {e}")
                        break

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
            if len(data) < 4:
                logger.warning("Data table packet too small")
                return
                
            num_tables = int.from_bytes(data[offset:offset+4], byteorder='little')
            logger.debug(f"Processing {num_tables} data tables")
            offset += 4

            for _ in range(num_tables):
                if offset + 8 > len(data):
                    logger.warning("Insufficient data for data table")
                    break
                    
                table_id = int.from_bytes(data[offset:offset+4], byteorder='little')
                num_entries = int.from_bytes(data[offset+4:offset+8], byteorder='little')
                offset += 8
                
                logger.debug(f"Data table {table_id}: {num_entries} entries")
                
                # Initialize table if needed
                if table_id not in self.data_tables:
                    self.data_tables[table_id] = {}
                    
                # Process entries
                for _ in range(num_entries):
                    if offset + 4 > len(data):
                        break
                        
                    # Read entry with both offset and table_id
                    entry = self._read_data_table_entry(data, offset, table_id)
                    if entry:
                        self.data_tables[table_id][entry.entry_id] = entry
                        offset += self._calculate_entry_size(entry)
                    else:
                        # If entry parsing failed, try to skip to next entry
                        logger.warning(f"Failed to parse entry in table {table_id}, skipping")
                        offset += 4  # Skip at least the entry ID
                        
            logger.debug(f"Finished processing {num_tables} data tables")

        except Exception as e:
            logger.error(f"Error updating data tables: {e}", exc_info=True)
    

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
                # First pass: try to parse header
                header = self._parse_demo_header(demo_file)
                if not header:
                    # If first pass fails, try analyzing file structure
                    logger.info("Initial header parse failed, analyzing file structure...")
                    self._analyze_file_structure(demo_file)
                    
                    # Reset file position and try again with more info
                    demo_file.seek(0)
                    header = self._parse_demo_header(demo_file)
                    
                    if not header:
                        raise DemoParserException("Could not parse demo header after multiple attempts")
                
                self.header = header
                logger.info(f"Successfully parsed header: format={header.format}, map={header.map_name}")
                
                # Based on format, choose appropriate packet parsing strategy
                if header.format == DemoFormat.PBDEMS2:
                    logger.info("Using PBDEMS2 packet parsing")
                    packets = self._parse_pbdems2_packets(demo_file)
                else:
                    logger.info("Using HL2DEMO packet parsing")
                    packets = self._parse_hl2demo_packets(demo_file)
                
                # Process the packets
                for packet in packets:
                    if packet.cmd_type in self._message_type_handlers:
                        try:
                            self._message_type_handlers[packet.cmd_type].handle_packet(packet)
                        except DemoParserException as e:
                            logger.warning(f"Error handling packet: {e}")
                            continue
                
                return {
                    'header': self.header.to_dict() if self.header else {},
                    'total_packets': len(packets),
                    'map': self.header.map_name if self.header else 'Unknown',
                    'server': self.header.server_name if self.header else 'Unknown',
                    'rounds': len(self.rounds),
                    'events': len(self.events)
                }
                
        except DemoParserException as e:
            logger.error(f"Demo parsing error: {e}")
            return {
                'error': str(e),
                'map': 'Unknown',
                'rounds': 0,
                'events': 0
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return {
                'error': f"Unexpected error: {str(e)}",
                'map': 'Unknown',
                'rounds': 0,
                'events': 0
            }
    
    def _parse_hl2demo_packets(self, demo_file: BinaryIO) -> List[DemoPacket]:
        """Parse packets from an HL2DEMO format demo file"""
        packets = []
        try:
            while True:
                packet = self._read_hl2demo_packet(demo_file)
                if packet is None:  # EOF or end of demo
                    break
                packets.append(packet)
                
                # Check for demo end
                if packet.cmd_type == DemoPacketType.DEM_STOP:
                    logger.info("Reached end of demo marker")
                    break
                    
        except PacketReadError as e:
            logger.warning(f"Error reading packet: {e}")
            # Try to resync and continue
            if self._resync_to_valid_packet(demo_file):
                logger.info("Successfully resynced after error")
        except Exception as e:
            logger.error(f"Error parsing HL2DEMO packets: {e}")
            raise DemoParserException(f"HL2DEMO packet parsing failed: {e}")
            
        return packets

    def _read_hl2demo_packet(self, demo_file: BinaryIO) -> Optional[DemoPacket]:
        """Read a single packet from an HL2DEMO demo file"""
        try:
            # Read command type (1 byte)
            cmd_type_data = demo_file.read(1)
            if not cmd_type_data:
                return None  # EOF
                
            cmd_type = cmd_type_data[0]
            
            # Read tick number (4 bytes)
            tick_data = demo_file.read(4)
            if not tick_data:
                raise PacketReadError("EOF while reading tick")
            tick = int.from_bytes(tick_data, byteorder='little')
            
            # Read packet slot (ignored in CS2, but need to skip)
            demo_file.read(1)
            
            # Read data size
            size_data = demo_file.read(4)
            if not size_data:
                raise PacketReadError("EOF while reading size")
            size = int.from_bytes(size_data, byteorder='little')
            
            # Validate packet
            if not self._is_valid_packet_header(cmd_type, tick, size):
                raise PacketReadError(f"Invalid packet header: type={cmd_type}, tick={tick}, size={size}")
                
            # Read packet data
            data = demo_file.read(size)
            if len(data) < size:
                raise PacketReadError(f"Incomplete packet: expected {size} bytes, got {len(data)}")
                
            return DemoPacket(cmd_type=cmd_type, tick=tick, data=data)
            
        except Exception as e:
            logger.error(f"Error reading HL2DEMO packet: {e}")
            raise PacketReadError(f"Failed to read HL2DEMO packet: {e}")

    def _parse_pbdems2_map_name(self, data: bytes) -> str:
        """Parse map name from PBDEMS2 header with stricter validation"""
        logger.debug("Starting map name parse")
        
        try:
            # Look for map prefixes with strict bounds
            map_prefixes = [b'de_', b'cs_', b'aim_', b'workshop/']
            
            # Find all potential map names
            potential_maps = []
            for prefix in map_prefixes:
                pos = data.find(prefix)
                while pos >= 0:
                    # Found a prefix, now find end using strict rules
                    end = pos
                    for i in range(pos, min(pos + 32, len(data))):
                        # Stop at first non-map character
                        if data[i] in b'\x00\r\n\t' or not (32 <= data[i] <= 126):
                            end = i
                            break
                    
                    potential_map = data[pos:end]
                    try:
                        map_str = potential_map.decode('ascii', errors='strict')
                        # Only accept valid map names
                        if all(c.isprintable() and c in '0123456789abcdefghijklmnopqrstuvwxyz_-' for c in map_str.lower()):
                            potential_maps.append(map_str)
                    except UnicodeDecodeError:
                        pass
                    
                    # Look for next occurrence
                    pos = data.find(prefix, end)
            
            logger.debug(f"Found potential maps: {potential_maps}")
            
            # Validate found maps
            COMMON_MAPS = {
                'de_dust2', 'de_mirage', 'de_inferno', 'de_nuke', 'de_overpass',
                'de_ancient', 'de_anubis', 'de_vertigo'
            }
            
            # First try to find a common map
            for map_name in potential_maps:
                if map_name.lower() in COMMON_MAPS:
                    logger.info(f"Found common map: {map_name}")
                    return map_name
            
            # If no common map, take first valid map
            for map_name in potential_maps:
                if len(map_name) >= 4 and len(map_name) <= 32:  # Reasonable map name length
                    logger.info(f"Found valid map name: {map_name}")
                    return map_name
            
            logger.warning("No valid map name found")
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error parsing map name: {e}")
            return "unknown"

    def _parse_pbdems2_packets(self, demo_file: BinaryIO) -> List[DemoPacket]:
        """Parse packets from a PBDEMS2 format demo file"""
        packets = []
        try:
            while True:
                packet = self._read_pbdems2_packet(demo_file)
                if packet is None:  # EOF or end of demo
                    break
                packets.append(packet)
                
                # Check for demo end
                if packet.cmd_type == DemoPacketType.DEM_STOP:
                    logger.info("Reached end of demo marker")
                    break
                    
        except PacketReadError as e:
            logger.warning(f"Error reading packet: {e}")
            # Try to resync and continue
            if self._resync_to_valid_packet(demo_file):
                logger.info("Successfully resynced after error")
        except Exception as e:
            logger.error(f"Error parsing PBDEMS2 packets: {e}")
            raise DemoParserException(f"PBDEMS2 packet parsing failed: {e}")
            
        return packets

    def _read_pbdems2_packet(self, demo_file: BinaryIO) -> Optional[DemoPacket]:
        """Read a single packet from a PBDEMS2 demo file with better marker detection"""
        try:
            # Read initial chunk for analysis
            initial_bytes = demo_file.read(16)
            if not initial_bytes:
                return None

            # Known packet markers and their corresponding offsets to the packet data
            PACKET_PATTERNS = [
                (b'\x07\xD0', 2),   # Common packet marker
                (b'\x01\xF1', 2),   # Alternative packet marker
                (b'\x20\x01', 2),   # Another known pattern
                (b'\x3C\x00', 0),   # Additional PBDEMS2 marker
                (b'\x40\x00', 0),   # Additional PBDEMS2 marker
            ]

            # Log initial bytes for debugging
            hex_str = ' '.join(f'{b:02x}' for b in initial_bytes[:4])
            logger.debug(f"Initial bytes: {hex_str}")

            # Try to find a valid marker
            found_pattern = None
            pattern_offset = 0

            for pattern, offset in PACKET_PATTERNS:
                if initial_bytes.startswith(pattern):
                    found_pattern = pattern
                    pattern_offset = offset
                    break

            if not found_pattern:
                # If no known pattern found, try PBDEMS2 command detection
                cmd_type = initial_bytes[0]
                if 0 <= cmd_type <= 32:  # Valid command range
                    pattern_offset = 0
                    found_pattern = initial_bytes[:1]
                else:
                    raise PacketReadError(f"Invalid packet marker: {initial_bytes[:4].hex()}")

            # Parse packet header after marker
            pos = pattern_offset
            cmd_type = initial_bytes[pos]
            
            # For PBDEMS2, tick and size are uint16 (2 bytes each)
            tick = int.from_bytes(initial_bytes[pos+1:pos+3], byteorder='little')
            size = int.from_bytes(initial_bytes[pos+3:pos+5], byteorder='little')

            logger.debug(f"Packet header: type={cmd_type}, tick={tick}, size={size}")

            # Validate packet
            if not self._is_valid_packet_size(size):
                raise PacketReadError(f"Invalid packet size: {size}")

            # Read packet data
            remaining_data = demo_file.read(size)
            if len(remaining_data) < size:
                raise PacketReadError(f"Incomplete packet: expected {size}, got {len(remaining_data)}")

            packet = DemoPacket(cmd_type=cmd_type, tick=tick, data=remaining_data)
            logger.debug(f"Successfully read packet: type={cmd_type}, tick={tick}, size={size}")
            return packet

        except Exception as e:
            logger.error(f"Error reading PBDEMS2 packet: {e}")
            raise PacketReadError(str(e))
        
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