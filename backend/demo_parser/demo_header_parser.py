# demo_header_parser.py

import struct
import logging
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class DemoHeader:
    format: str
    demo_protocol: int
    network_protocol: int
    server_name: str
    map_name: str
    client_name: str
    game_directory: str
    playback_time: float = 0.0
    ticks: int = 0
    frames: int = 0
    sign_on_length: int = 0

class DemoHeaderParser:  # Note this exact name
    # Known CS2 maps
    KNOWN_MAPS = {
        'de_mirage', 'de_dust2', 'de_inferno', 'de_nuke', 'de_overpass',
        'de_ancient', 'de_vertigo', 'de_anubis'
    }

    def __init__(self, demo_path: str):
        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
        self.demo_path = demo_path
        self.header = None

    def _extract_map_name(self, data: bytes) -> str:
        """Extract clean map name from data"""
        try:
            pos = data.find(b'de_')
            if pos >= 0:
                end = pos
                for i in range(pos, min(pos + 32, len(data))):
                    if data[i] in b'!/ \t\n\r':
                        end = i
                        break
                
                raw_map = data[pos:end].decode('ascii', errors='replace')
                map_name = raw_map.lower()
                
                for known_map in self.KNOWN_MAPS:
                    if map_name.startswith(known_map):
                        return known_map
                
                base_map = re.sub(r'\d+$', '', map_name)
                return base_map
            
            return "unknown"
        except Exception as e:
            logger.error(f"Error extracting map name: {e}")
            return "unknown"

    def _extract_server_info(self, data: bytes) -> str:
        """Extract server information"""
        try:
            if b'Valve Counter-Strike 2' in data:
                server_region = ""
                if b'us_southwest' in data:
                    server_region = "US Southwest"
                elif b'us_east' in data:
                    server_region = "US East"
                else:
                    server_region = "Valve"
                return f"Valve CS2 {server_region} Server"
            elif b'FACEIT' in data:
                return "FACEIT Server"
            elif b'ESL' in data:
                return "ESL Server"
            
            return "CS2 Server"
        except Exception as e:
            logger.error(f"Error extracting server info: {e}")
            return "CS2 Server"

    def parse(self) -> Dict[str, Any]:
        """Parse the demo header"""
        try:
            with open(self.demo_path, 'rb') as demo_file:
                # Read and validate magic bytes
                magic = demo_file.read(8)
                if not magic.startswith(b'PBDEMS'):
                    raise ValueError("Not a PBDEMS demo file")

                # Read protocol versions
                demo_protocol = struct.unpack('i', demo_file.read(4))[0]
                network_protocol = struct.unpack('i', demo_file.read(4))[0]

                # Read chunk for analysis
                demo_file.seek(0)
                data_chunk = demo_file.read(4096)

                # Extract information
                map_name = self._extract_map_name(data_chunk)
                server_name = self._extract_server_info(data_chunk)

                self.header = DemoHeader(
                    format="PBDEMS2",
                    demo_protocol=demo_protocol,
                    network_protocol=network_protocol,
                    server_name=server_name,
                    map_name=map_name,
                    client_name="",
                    game_directory="csgo",
                    playback_time=0.0,
                    ticks=0,
                    frames=0,
                    sign_on_length=0
                )

                return self.header.__dict__

        except Exception as e:
            logger.error(f"Error parsing demo header: {e}")
            raise