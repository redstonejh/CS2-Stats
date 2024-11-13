# demo_packet_parser.py

import struct
from typing import Dict, Any, Optional, BinaryIO, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DemoPacket:
    """Represents a parsed demo packet"""
    cmd_type: int
    tick: int
    data: bytes

class DemoPacketParser:
    """Parser for CS2/PBDEMS2 demo packets"""
    
    # Protocol markers
    PBDM_MARKER = b'PBDM'
    CS2_MARKERS = [b'\x07\xD0', b'\x01\xF1']
    
    PACKET_TYPES = {
    0x00: "EmptyPacket",
    0x02: "PacketEntity",
    0x04: "StringTable",
    0x05: "CreateStringTable",
    0x06: "UserMessage", 
    0x08: "DataTable",
    0x10: "ParseTables",
    0x18: "InstanceBaseline",
    0x30: "ConsoleCmd",
    0x40: "CustomData",
    0x92: "SyncTick",
    0xBF: "SignOnState",
    0xC5: "GameEvent"
}
    
    def _parse_game_event(self, data: bytes) -> Dict[str, Any]:
        """Parse a game event packet"""
        return {
            'event_type': 'GameEvent',
            'data_size': len(data)
        }
    
    def _parse_packet_entity(self, data: bytes) -> Dict[str, Any]:
        """Parse an entity update packet"""
        return {
            'entity_type': 'PacketEntity',
            'data_size': len(data)
        }

    def _update_packet_info(self, packet_info: Dict[str, Any], packet: DemoPacket):
        """Add type-specific information to packet info"""
        if packet.cmd_type == 0x01:  # GameEvent
            packet_info.update(self._parse_game_event(packet.data))
        elif packet.cmd_type == 0x02:  # PacketEntity
            packet_info.update(self._parse_packet_entity(packet.data))

    
    def __init__(self, demo_file: BinaryIO, format_type: str = "PBDEMS2"):
        self.demo_file = demo_file
        self.format_type = format_type
        logger.debug("Initializing packet parser")
        self.initialize_parser()
    
    def _debug_bytes(self, data: bytes, description: str, preview_size: int = 32):
        """Debug helper to print byte content"""
        preview = data[:preview_size]
        hex_str = ' '.join(f'{b:02x}' for b in preview)
        ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in preview)
        logger.debug(f"{description} - Hex: {hex_str}, ASCII: {ascii_str}")

    def initialize_parser(self):
        """Initialize the parser and find the start of packet data"""
        logger.debug("Starting parser initialization")
        
        # Reset to start of file
        self.demo_file.seek(0)
        
        # Skip header
        self.demo_file.seek(1072)  # Standard header size
        
        # Look for first PBDM marker
        self._find_next_marker()
        logger.debug(f"Initialized parser at position {self.demo_file.tell()}")

    def _find_next_marker(self) -> bool:
        """Find the next valid packet marker"""
        buffer = bytearray()
        while True:
            byte = self.demo_file.read(1)
            if not byte:
                return False
                
            buffer.extend(byte)
            
            # Keep buffer at reasonable size
            if len(buffer) > 8:
                buffer = buffer[-8:]
            
            # Look for PBDM marker
            if self.PBDM_MARKER in buffer:
                self.demo_file.seek(-(len(buffer) - buffer.find(self.PBDM_MARKER)), 1)
                logger.debug(f"Found PBDM marker at {self.demo_file.tell()}")
                return True
                
            # Look for CS2 markers
            for marker in self.CS2_MARKERS:
                if marker in buffer:
                    self.demo_file.seek(-(len(buffer) - buffer.find(marker)), 1)
                    logger.debug(f"Found CS2 marker at {self.demo_file.tell()}")
                    return True
    
    def read_packet(self) -> Optional[DemoPacket]:
        """Read a packet from the demo file"""
        try:
            # Read 4 bytes to check for PBDM marker
            marker = self.demo_file.read(4)
            if not marker:
                return None
                
            self._debug_bytes(marker, "Potential marker")
            
            if marker == self.PBDM_MARKER:
                # Skip 4 bytes after PBDM marker
                self.demo_file.read(4)
                
                # Read packet type and size
                header = self.demo_file.read(2)
                if len(header) < 2:
                    return None
                    
                packet_type = header[0]
                size = header[1]
                
                logger.debug(f"PBDM packet - Type: {packet_type}, Size: {size}")
                
                # Read packet data
                data = self.demo_file.read(size)
                if len(data) < size:
                    return None
                    
                return DemoPacket(cmd_type=packet_type, tick=0, data=data)
                
            # Check for CS2 markers
            elif marker[:2] in self.CS2_MARKERS:
                # Read packet type and size
                header = marker[2:4]
                if len(header) < 2:
                    return None
                    
                packet_type = header[0]
                size = header[1]
                
                logger.debug(f"CS2 packet - Type: {packet_type}, Size: {size}")
                
                # Read packet data
                data = self.demo_file.read(size)
                if len(data) < size:
                    return None
                    
                return DemoPacket(cmd_type=packet_type, tick=0, data=data)
            
            # No valid marker found, try to find next marker
            self.demo_file.seek(-3, 1)  # Back up 3 bytes for overlapping search
            if not self._find_next_marker():
                return None
                
            return self.read_packet()
            
        except Exception as e:
            logger.error(f"Error reading packet: {e}", exc_info=True)
            return None
            
    def process_packets(self) -> List[Dict[str, Any]]:
        """Process all packets in the demo file"""
        packets = []
        packet_count = 0
        
        logger.debug("Starting packet processing")
        
        while True:
            current_pos = self.demo_file.tell()
            logger.debug(f"Reading packet at position {current_pos}")
            
            packet = self.read_packet()
            if not packet:
                continue  # Skip invalid packets but keep reading
                
            # Add packet to our list regardless of type (unless it's a stop packet)
            if packet.cmd_type == 0x02:  # dem_stop
                logger.info("Reached dem_stop packet")
                break
                
            packet_info = {
                'type': self.PACKET_TYPES.get(packet.cmd_type, f'unknown_{packet.cmd_type:02x}'),
                'tick': packet.tick,
                'size': len(packet.data)
            }
            
            # Add packet to our list
            packets.append(packet_info)
            packet_count += 1
            
            if packet_count % 100 == 0:
                logger.info(f"Processed {packet_count} packets...")
        
        logger.info(f"Finished processing {packet_count} packets")
        return packets