# demo_analyzer.py

import logging
import sys
import os
from typing import Dict, Any, List
from demo_header_parser import DemoHeaderParser
from demo_packet_parser import DemoPacketParser

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Show all debug messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoAnalyzer:
    def __init__(self, demo_path: str):
        """Initialize analyzer with demo file path"""
        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"Demo file not found: {demo_path}")
            
        self.demo_path = demo_path
        self.header = None
        logger.debug(f"Initialized analyzer for {demo_path}")
        
    def parse_header(self) -> Dict[str, Any]:
        """Parse the demo header first"""
        try:
            logger.info("Parsing demo header...")
            header_parser = DemoHeaderParser(self.demo_path)
            self.header = header_parser.parse()
            logger.debug(f"Parsed header: {self.header}")
            return self.header
        except Exception as e:
            logger.error(f"Error parsing header: {e}", exc_info=True)
            raise

    def parse_packets(self) -> List[Dict[str, Any]]:
        """Parse packets from the demo file"""
        try:
            # Make sure we've parsed the header first
            if not self.header:
                self.parse_header()
            
            logger.debug("Opening demo file for packet parsing")
            with open(self.demo_path, 'rb') as demo_file:
                # Create packet parser
                packet_parser = DemoPacketParser(demo_file, self.header['format'])
                logger.debug("Created packet parser")
                
                # Process packets
                logger.debug("Starting packet processing")
                packets = packet_parser.process_packets()
                
                logger.info(f"Parsed {len(packets)} packets from demo")
                logger.debug(f"First few packets: {packets[:5] if packets else 'None'}")
                
                return packets
                
        except Exception as e:
            logger.error(f"Error parsing packets: {e}", exc_info=True)
            raise

    def analyze(self) -> Dict[str, Any]:
        """Analyze the demo file"""
        try:
            logger.debug("Starting demo analysis")
            
            # Parse header
            self.parse_header()
            
            # Parse packets
            logger.info("Parsing packets...")
            packets = self.parse_packets()
            
            analysis_result = {
                "header": self.header,
                "packets": packets,
                "analysis": {
                    "map": self.header["map_name"],
                    "server": self.header["server_name"],
                    "total_packets": len(packets)
                }
            }
            
            logger.debug("Analysis complete")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            raise

def print_analysis_results(results: Dict[str, Any]) -> None:
    """Print analysis results in a formatted way"""
    print("\nDemo Analysis Results")
    print("=" * 50)
    
    # Header Information
    print("\nHeader Information:")
    print("-" * 20)
    print(f"Format: {results['header']['format']}")
    print(f"Map: {results['header']['map_name']}")
    print(f"Server: {results['header']['server_name']}")
    print(f"Demo Protocol: {results['header']['demo_protocol']}")
    print(f"Network Protocol: {results['header']['network_protocol']}")
    
    # Packet Information
    print("\nPacket Information:")
    print("-" * 20)
    print(f"Total Packets: {results['analysis']['total_packets']}")
    
    # Show first few packets if available
    if results['packets']:
        print("\nFirst few packets:")
        for i, packet in enumerate(results['packets'][:5]):
            print(f"Packet {i+1}: {packet}")

def main():
    print("\nCS2 Demo Analyzer")
    print("=" * 50)
    
    if len(sys.argv) != 2:
        print("\nUsage: python demo_analyzer.py <demo_file.dem>")
        sys.exit(1)
    
    demo_path = sys.argv[1]
    
    try:
        logger.debug(f"Starting analysis of {demo_path}")
        analyzer = DemoAnalyzer(demo_path)
        results = analyzer.analyze()
        print_analysis_results(results)
        
    except Exception as e:
        print(f"\nError analyzing demo: {e}")
        logger.error("Analysis failed", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()