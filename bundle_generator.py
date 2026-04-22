#!/usr/bin/env python3
"""
Enterprise Support Bundle Generator
Generates realistic HPE AOS-CX switch support bundles with injected failures
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml


class TimestampGenerator:
    """Generates sequential timestamps for log consistency"""
    
    def __init__(self, start_time: datetime, timezone: str = "+05:30"):
        self.current = start_time
        self.timezone = timezone
    
    def next(self, min_delta_ms: int = 10, max_delta_ms: int = 5000) -> str:
        """Generate next timestamp with random microsecond increment"""
        delta = random.randint(min_delta_ms, max_delta_ms)
        self.current += timedelta(milliseconds=delta)
        # Format: 2026-02-25T11:00:23.351987+05:30
        return self.current.strftime(f"%Y-%m-%dT%H:%M:%S.%f{self.timezone}")
    
    def get_showtech_time(self) -> str:
        """Format for show tech header"""
        return self.current.strftime("%a %b %d %H:%M:%S %Y")
    
    def get_current(self) -> datetime:
        return self.current


class LogSimulator:
    """Generates realistic messages.log entries"""
    
    def __init__(self, hostname: str, ts_gen: TimestampGenerator):
        self.hostname = hostname
        self.ts = ts_gen
        self.processes = {
            "acctsyslogd": 387,
            "hpe-routing": 4714,
            "kernel": None,
            "ops-fand": 512,
            "ops-ledd": 523,
            "vland": 489
        }
    
    def generate_rest_audit(self, user: str = "afc_admin", 
                           source_ip: str = "10.233.131.22") -> str:
        """Generate REST API audit log"""
        endpoints = [
            "/rest/v10.16/system/snmpv3_users?selector=configuration&depth=1",
            "/rest/v10.16/system/snmp_traps?selector=configuration&depth=1",
            "/rest/v10.16/system?selector=writable",
            "/rest/v10.16/system/vrfs/mgmt?selector=configuration&depth=1",
            "/rest/v10.16/system/bgp_aspath_filters?depth=2&selector=configuration",
            "/rest/v10.16/system/interfaces?depth=2&selector=status"
        ]
        uri = random.choice(endpoints)
        
        return (f"{self.ts.next()} {self.hostname} acctsyslogd[{self.processes['acctsyslogd']}]: "
                f"AUDIT|REST URI executed by user '{user}' from address '{source_ip}' "
                f"through REST session. The URI is '{uri}' and the method is GET "
                f"and response code is 200 with payload \"\" which resulted in success "
                f"at timezone Asia/Calcutta.")
    
    def generate_kernel_message(self, message: str, level: str = "INFO") -> str:
        """Generate kernel log message"""
        return f"{self.ts.next()} {self.hostname} kernel: {message}"
    
    def generate_routing_log(self, severity: str, subsystem: str, 
                            seqno: int, message: str) -> str:
        """Generate hpe-routing daemon log"""
        return (f"{self.ts.next()} {self.hostname} hpe-routing[{self.processes['hpe-routing']}]: "
                f"ovs|{seqno}|{subsystem}|{severity}|{message}")
    
    def generate_interface_flap(self, interface: str) -> List[str]:
        """Generate interface flap event sequence"""
        logs = []
        logs.append(self.generate_kernel_message(
            f"provision 0000:04:00.0 {interface}: NIC Link is Down"))
        logs.append(self.generate_routing_log(
            "ERR", "interface_mgr", random.randint(10000, 20000),
            f"Interface {interface} state changed to down"))
        # Brief delay before recovery attempt
        logs.append(self.generate_kernel_message(
            f"provision 0000:04:00.0 {interface}: NIC Link is Up 25000 Mbps Full Duplex"))
        logs.append(self.generate_routing_log(
            "INFO", "interface_mgr", random.randint(10000, 20000),
            f"Interface {interface} state changed to up"))
        return logs
    
    def generate_bgp_neighbor_down(self, neighbor_ip: str) -> List[str]:
        """Generate BGP neighbor down event"""
        logs = []
        logs.append(self.generate_routing_log(
            "WARN", "bgp", random.randint(10000, 20000),
            f"BGP peer {neighbor_ip} connection lost"))
        logs.append(self.generate_routing_log(
            "ERR", "bgp", random.randint(10000, 20000),
            f"BGP session to {neighbor_ip} went down - Hold Timer Expired"))
        logs.append(self.generate_routing_log(
            "INFO", "bgp", random.randint(10000, 20000),
            f"BGP peer {neighbor_ip} attempting reconnection"))
        return logs
    
    def generate_vlan_mismatch(self, interface: str, expected_vlan: int, 
                               received_vlan: int) -> List[str]:
        """Generate VLAN mismatch error"""
        logs = []
        logs.append(self.generate_routing_log(
            "ERR", "vlan_mgr", random.randint(10000, 20000),
            f"VLAN mismatch on {interface}: expected {expected_vlan}, received {received_vlan}"))
        logs.append(self.generate_kernel_message(
            f"{interface}: dropping frame with unexpected VLAN tag {received_vlan}"))
        return logs
    
    def generate_route_missing(self, destination: str) -> List[str]:
        """Generate missing route error"""
        logs = []
        logs.append(self.generate_routing_log(
            "ERR", "route_mgr", random.randint(10000, 20000),
            f"Route to {destination} not found in routing table"))
        logs.append(self.generate_routing_log(
            "WARN", "route_mgr", random.randint(10000, 20000),
            f"Packet destined to {destination} dropped - no route to host"))
        return logs
    
    def generate_noise_burst(self, count: int = 10) -> List[str]:
        """Generate background noise logs"""
        logs = []
        for _ in range(count):
            log_type = random.choice(["rest", "kernel_ratelimit", "napi"])
            if log_type == "rest":
                logs.append(self.generate_rest_audit())
            elif log_type == "kernel_ratelimit":
                logs.append(self.generate_kernel_message(
                    f"net_ratelimit: {random.randint(50, 200)} callbacks suppressed"))
            else:
                interface = random.choice(["pvnet0", "pvnet1"])
                logs.append(self.generate_kernel_message(
                    f"provision 0000:05:00.0 {interface}: Re-enabling RX interrupts and disabling NAPI polling"))
        return logs


class ShowTechSimulator:
    """Generates realistic showtech.txt command outputs"""
    
    def __init__(self, hostname: str, ts_gen: TimestampGenerator):
        self.hostname = hostname
        self.ts = ts_gen
    
    def generate_header(self) -> str:
        """Generate show tech header"""
        time_str = self.ts.get_showtech_time()
        return f"""====================================================
Show Tech executed on {time_str}
====================================================
====================================================
[Begin] Feature basic
====================================================

"""
    
    def generate_clock(self) -> str:
        """Generate show clock output"""
        time_str = self.ts.get_showtech_time()
        return f"""
*********************************
Command : show clock
*********************************
{time_str}
System is configured for timezone : Asia/Calcutta
"""
    
    def generate_version(self, version: str = "LL.10.16.1010") -> str:
        """Generate show version output"""
        return f"""
*********************************
Command : show version
*********************************
-----------------------------------------------------------------------------
AOS-CX
(c) Copyright 2017-2025 Hewlett Packard Enterprise Development LP
-----------------------------------------------------------------------------
Version      : {version}
Build Date   : 2025-11-10 17:20:26 UTC
Build ID     : AOS-CX:{version}:8e02d0e3b736:202511101706
Build SHA    : 8e02d0e3b73610a4383330daa86e9bddc19c3283
Hot Patches  :
Active Image : secondary

Service OS Version : LL.01.17.0001
BIOS Version       : LL.01.0002
"""
    
    def generate_interface_dom(self, interfaces: Dict[str, dict]) -> str:
        """Generate show interface dom output"""
        output = """
*********************************
Command : show interface dom
*********************************
----------------------------------------------------------------------------
Port      Type             Lane     Temp  Voltage  Tx Bias Rx Power Tx Power
                                     (C)      (V)     (mA)    (dBm)    (dBm)
----------------------------------------------------------------------------
"""
        for port, state in interfaces.items():
            if state.get("transceiver_type"):
                temp = state.get("temp", round(random.uniform(40, 55), 2))
                voltage = state.get("voltage", round(random.uniform(3.25, 3.35), 2))
                tx_bias = state.get("tx_bias", round(random.uniform(7.2, 7.5), 2))
                
                # Simulate failure conditions
                if state.get("link_down", False):
                    rx_power = -40.00  # No signal
                    tx_power = -40.00
                else:
                    rx_power = round(random.uniform(-3, -0.5), 2)
                    tx_power = round(random.uniform(-2.5, -0.5), 2)
                
                output += f"{port:<10}{state['transceiver_type']:<17}      {temp:>6.2f}   {voltage:>6.2f}   {tx_bias:>6.2f}   {rx_power:>6.2f}   {tx_power:>6.2f}\n"
        
        return output
    
    def generate_interface_brief(self, interfaces: Dict[str, dict]) -> str:
        """Generate show interface brief output"""
        output = """
*********************************
Command : show interface brief
*********************************
--------------------------------------------------------------------------------
Port       Native  Mode    Type       Enabled  Status  Reason         Speed
           VLAN                                                       
--------------------------------------------------------------------------------
"""
        for port, state in interfaces.items():
            vlan = state.get("vlan", "-")
            mode = state.get("mode", "access")
            itype = state.get("type", "25G-SR")
            enabled = "yes" if state.get("enabled", True) else "no"
            status = "up" if state.get("link_up", True) and not state.get("link_down", False) else "down"
            reason = state.get("down_reason", "")
            speed = state.get("speed", "25G") if status == "up" else ""
            
            output += f"{port:<11}{vlan:<8}{mode:<8}{itype:<11}{enabled:<9}{status:<8}{reason:<15}{speed}\n"
        
        return output


class RouteInfoSimulator:
    """Generates realistic routeinfo.txt command outputs"""
    
    def __init__(self, vrf: str = "VRF_1"):
        self.vrf = vrf
    
    def generate_header(self, command: str) -> str:
        """Generate command header"""
        return f"""-------------------------------------------------------------
/usr/bin/sudo ip netns exec {self.vrf} {command}
-------------------------------------------------------------
"""
    
    def generate_neighbor_show(self, neighbors: List[Dict]) -> str:
        """Generate ip neighbor show output"""
        output = self.generate_header("ip -4 neighbor show")
        for n in neighbors:
            state = n.get("state", "REACHABLE")
            output += f"{n['ip']} dev {n['interface']} lladdr {n['mac']} {state}\n"
        return output
    
    def generate_route_show(self, routes: List[Dict], local_ip: str) -> str:
        """Generate ip route show table all output"""
        output = self.generate_header("ip -4 route show table all")
        
        for r in routes:
            if r["type"] == "default":
                output += f"default via {r['via']} dev {r['dev']} proto static\n"
            elif r["type"] == "connected":
                output += f"{r['network']} dev {r['dev']} proto kernel scope link src {local_ip}\n"
            elif r["type"] == "local":
                output += f"local {r['ip']} dev {r['dev']} table local proto kernel scope host src {r['ip']}\n"
        
        return output
    
    def generate_interface_stats(self, interfaces: List[Dict]) -> str:
        """Generate ip -s -s link show output"""
        output = self.generate_header("ip -s -s link show")
        
        for idx, iface in enumerate(interfaces, start=1):
            state = "UP" if iface.get("up", True) else "DOWN"
            flags = f"<BROADCAST,MULTICAST,{'UP,LOWER_UP' if state == 'UP' else 'DOWN'}>"
            
            rx_bytes = iface.get("rx_bytes", random.randint(100000000, 500000000))
            rx_packets = iface.get("rx_packets", random.randint(500000, 1000000))
            rx_errors = iface.get("rx_errors", 0)
            rx_dropped = iface.get("rx_dropped", 0)
            
            tx_bytes = iface.get("tx_bytes", random.randint(1000000000, 5000000000))
            tx_packets = iface.get("tx_packets", random.randint(1000000, 3000000))
            tx_errors = iface.get("tx_errors", 0)
            tx_dropped = iface.get("tx_dropped", 0)
            
            output += f"""{idx}: {iface['name']}: {flags} mtu {iface.get('mtu', 1500)} qdisc mq state {state} mode DEFAULT group default qlen {iface.get('qlen', 1000)}
    link/ether {iface.get('mac', '00:00:00:00:00:00')} brd ff:ff:ff:ff:ff:ff
    RX: bytes  packets  errors  dropped missed  mcast   
    {rx_bytes:<11}{rx_packets:<9}{rx_errors:<8}{rx_dropped:<8}0       0       
    RX errors: length   crc     frame   fifo    overrun
               0        0       0       0       0       
    TX: bytes  packets  errors  dropped carrier collsns 
    {tx_bytes:<11}{tx_packets:<9}{tx_errors:<8}{tx_dropped:<8}0       0       
    TX errors: aborted  fifo   window heartbeat transns
               0        0       0       0       0       
"""
        return output


class FailureScenario:
    """Base class for failure scenarios"""
    
    def __init__(self, name: str, description: str, severity: str):
        self.name = name
        self.description = description
        self.severity = severity
        self.failure_time: Optional[datetime] = None
        self.root_cause_entity: Optional[str] = None
        self.root_cause_type: Optional[str] = None
        self.symptoms: List[str] = []
        self.recommended_action: Optional[str] = None
    
    def inject_into_logs(self, log_sim: LogSimulator) -> List[str]:
        """Override in subclass to inject failure logs"""
        raise NotImplementedError
    
    def modify_showtech(self, showtech_sim: ShowTechSimulator, 
                       interfaces: Dict) -> Dict:
        """Override in subclass to modify interface state"""
        return interfaces
    
    def modify_routeinfo(self, route_sim: RouteInfoSimulator,
                        neighbors: List, routes: List, 
                        ifaces: List) -> Tuple[List, List, List]:
        """Override in subclass to modify routing state"""
        return neighbors, routes, ifaces
    
    def get_ground_truth(self) -> Dict:
        """Return ground truth metadata for evaluation"""
        return {
            "scenario_name": self.name,
            "description": self.description,
            "severity": self.severity,
            "failure_timestamp": self.failure_time.isoformat() if self.failure_time else None,
            "root_cause_entity": self.root_cause_entity,
            "root_cause_type": self.root_cause_type,
            "expected_symptoms": self.symptoms,
            "recommended_action": self.recommended_action
        }


class PortFlapScenario(FailureScenario):
    """Interface flapping due to physical layer issues"""
    
    def __init__(self, interface: str = "1/1/3"):
        super().__init__(
            name="port_flap",
            description=f"Interface {interface} experiencing link flapping",
            severity="high"
        )
        self.interface = interface
        self.root_cause_entity = interface
        self.root_cause_type = "physical_layer_failure"
        self.symptoms = ["link_state_change", "packet_loss", "connection_timeout"]
        self.recommended_action = f"Check physical cable and transceiver on {interface}"
    
    def inject_into_logs(self, log_sim: LogSimulator) -> List[str]:
        logs = []
        self.failure_time = log_sim.ts.get_current()
        
        # Generate multiple flaps
        for _ in range(3):
            logs.extend(log_sim.generate_noise_burst(5))
            logs.extend(log_sim.generate_interface_flap(self.interface))
            logs.extend(log_sim.generate_noise_burst(3))
        
        return logs
    
    def modify_showtech(self, showtech_sim: ShowTechSimulator, 
                       interfaces: Dict) -> Dict:
        if self.interface in interfaces:
            # Show degraded optical signal
            interfaces[self.interface]["rx_power"] = -40.00
            interfaces[self.interface]["tx_power"] = -40.00
            interfaces[self.interface]["link_down"] = True
            interfaces[self.interface]["down_reason"] = "Link flapping"
        return interfaces


class BGPNeighborDownScenario(FailureScenario):
    """BGP neighbor session failure"""
    
    def __init__(self, neighbor_ip: str = "10.233.255.1"):
        super().__init__(
            name="bgp_neighbor_down",
            description=f"BGP session to {neighbor_ip} failed",
            severity="critical"
        )
        self.neighbor_ip = neighbor_ip
        self.root_cause_entity = neighbor_ip
        self.root_cause_type = "routing_protocol_failure"
        self.symptoms = ["bgp_session_down", "route_withdrawal", "connectivity_loss"]
        self.recommended_action = f"Verify BGP configuration and network connectivity to {neighbor_ip}"
    
    def inject_into_logs(self, log_sim: LogSimulator) -> List[str]:
        logs = []
        self.failure_time = log_sim.ts.get_current()
        
        logs.extend(log_sim.generate_noise_burst(10))
        logs.extend(log_sim.generate_bgp_neighbor_down(self.neighbor_ip))
        logs.extend(log_sim.generate_noise_burst(5))
        
        return logs
    
    def modify_routeinfo(self, route_sim: RouteInfoSimulator,
                        neighbors: List, routes: List, 
                        ifaces: List) -> Tuple[List, List, List]:
        # Remove neighbor from ARP table
        neighbors = [n for n in neighbors if n["ip"] != self.neighbor_ip]
        # Remove default route
        routes = [r for r in routes if not (r.get("type") == "default" and 
                                            r.get("via") == self.neighbor_ip)]
        return neighbors, routes, ifaces


class VLANMismatchScenario(FailureScenario):
    """VLAN configuration mismatch"""
    
    def __init__(self, interface: str = "1/1/5", expected_vlan: int = 100, 
                 received_vlan: int = 200):
        super().__init__(
            name="vlan_mismatch",
            description=f"VLAN mismatch on {interface}: expected {expected_vlan}, got {received_vlan}",
            severity="medium"
        )
        self.interface = interface
        self.expected_vlan = expected_vlan
        self.received_vlan = received_vlan
        self.root_cause_entity = interface
        self.root_cause_type = "configuration_error"
        self.symptoms = ["frame_drops", "vlan_violation", "connectivity_partial"]
        self.recommended_action = f"Verify VLAN configuration on {interface} and connected device"
    
    def inject_into_logs(self, log_sim: LogSimulator) -> List[str]:
        logs = []
        self.failure_time = log_sim.ts.get_current()
        
        logs.extend(log_sim.generate_noise_burst(8))
        logs.extend(log_sim.generate_vlan_mismatch(self.interface, 
                                                   self.expected_vlan,
                                                   self.received_vlan))
        logs.extend(log_sim.generate_noise_burst(4))
        
        return logs


class MissingRouteScenario(FailureScenario):
    """Missing route in routing table"""
    
    def __init__(self, destination: str = "192.168.100.0/24"):
        super().__init__(
            name="missing_route",
            description=f"Route to {destination} missing from routing table",
            severity="high"
        )
        self.destination = destination
        self.root_cause_entity = destination
        self.root_cause_type = "routing_table_error"
        self.symptoms = ["packet_drops", "unreachable_destination", "icmp_host_unreachable"]
        self.recommended_action = f"Add static route or verify dynamic routing protocol for {destination}"
    
    def inject_into_logs(self, log_sim: LogSimulator) -> List[str]:
        logs = []
        self.failure_time = log_sim.ts.get_current()
        
        logs.extend(log_sim.generate_noise_burst(6))
        logs.extend(log_sim.generate_route_missing(self.destination))
        logs.extend(log_sim.generate_noise_burst(8))
        
        return logs


class SupportBundleGenerator:
    """Main orchestrator for generating complete support bundles"""
    
    SCENARIOS = {
        "port_flap": PortFlapScenario,
        "bgp_neighbor_down": BGPNeighborDownScenario,
        "vlan_mismatch": VLANMismatchScenario,
        "missing_route": MissingRouteScenario
    }
    
    def __init__(self, output_dir: Path, hostname: str = "PSCSCTLEAFB03"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hostname = hostname
    
    def generate(self, scenario_name: str, duration_minutes: int = 60,
                noise_level: str = "medium", **scenario_kwargs):
        """Generate a complete support bundle with injected failure"""
        
        # Initialize scenario
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario_class = self.SCENARIOS[scenario_name]
        scenario = scenario_class(**scenario_kwargs)
        
        # Initialize timestamp generator
        start_time = datetime.now().replace(microsecond=0)
        ts_gen = TimestampGenerator(start_time)
        
        # Initialize simulators
        log_sim = LogSimulator(self.hostname, ts_gen)
        showtech_sim = ShowTechSimulator(self.hostname, ts_gen)
        route_sim = RouteInfoSimulator()
        
        # Generate baseline topology
        interfaces = self._create_baseline_interfaces()
        neighbors = self._create_baseline_neighbors()
        routes = self._create_baseline_routes()
        ifaces = self._create_baseline_ifaces()
        
        # Generate logs with failure injection
        logs = self._generate_logs(log_sim, scenario, duration_minutes, noise_level)
        
        # Modify state based on failure
        interfaces = scenario.modify_showtech(showtech_sim, interfaces)
        neighbors, routes, ifaces = scenario.modify_routeinfo(route_sim, neighbors, 
                                                              routes, ifaces)
        
        # Write files
        self._write_messages_log(logs)
        self._write_showtech(showtech_sim, interfaces)
        self._write_routeinfo(route_sim, neighbors, routes, ifaces)
        self._write_metadata(scenario)
        
        print(f"✓ Support bundle generated: {self.output_dir}")
        print(f"  - Scenario: {scenario.name}")
        print(f"  - Severity: {scenario.severity}")
        print(f"  - Log lines: {len(logs)}")
        print(f"  - Duration: {duration_minutes} minutes")
    
    def _generate_logs(self, log_sim: LogSimulator, scenario: FailureScenario,
                      duration_minutes: int, noise_level: str) -> List[str]:
        """Generate log entries with failure injection"""
        logs = []
        
        # Noise configuration
        noise_config = {
            "low": {"burst_interval": 60, "burst_size": 5},
            "medium": {"burst_interval": 30, "burst_size": 10},
            "high": {"burst_interval": 15, "burst_size": 20}
        }
        config = noise_config.get(noise_level, noise_config["medium"])
        
        # Pre-failure normal operation
        pre_failure_duration = duration_minutes * 0.3  # 30% before failure
        for _ in range(int(pre_failure_duration * 60 / config["burst_interval"])):
            logs.extend(log_sim.generate_noise_burst(config["burst_size"]))
        
        # Inject failure
        failure_logs = scenario.inject_into_logs(log_sim)
        logs.extend(failure_logs)
        
        # Post-failure logs
        post_failure_duration = duration_minutes * 0.7
        for _ in range(int(post_failure_duration * 60 / config["burst_interval"])):
            logs.extend(log_sim.generate_noise_burst(config["burst_size"]))
        
        return logs
    
    def _create_baseline_interfaces(self) -> Dict[str, dict]:
        """Create baseline interface configuration"""
        interfaces = {}
        
        # 25G interfaces
        for i in range(1, 15):
            interfaces[f"1/1/{i}"] = {
                "transceiver_type": "25G-SR",
                "vlan": "100" if i <= 7 else "200",
                "mode": "access",
                "type": "25G-SR",
                "enabled": True,
                "link_up": True,
                "speed": "25G"
            }
        
        # 100G uplinks
        for i in range(49, 55):
            interfaces[f"1/1/{i}"] = {
                "transceiver_type": "100G-SR4",
                "mode": "trunk",
                "type": "100G-SR4",
                "enabled": True,
                "link_up": True,
                "speed": "100G"
            }
        
        return interfaces
    
    def _create_baseline_neighbors(self) -> List[Dict]:
        """Create baseline ARP neighbors"""
        return [
            {
                "ip": "10.233.255.1",
                "interface": "eth0",
                "mac": "02:02:00:bb:cc:01",
                "state": "REACHABLE"
            }
        ]
    
    def _create_baseline_routes(self) -> List[Dict]:
        """Create baseline routing table"""
        return [
            {"type": "default", "via": "10.233.255.1", "dev": "eth0"},
            {"type": "connected", "network": "10.233.255.0/24", "dev": "eth0"},
            {"type": "local", "ip": "10.233.255.91", "dev": "eth0"}
        ]
    
    def _create_baseline_ifaces(self) -> List[Dict]:
        """Create baseline interface statistics"""
        return [
            {
                "name": "lo",
                "mac": "00:00:00:00:00:00",
                "up": True,
                "mtu": 65536,
                "qlen": 1000
            },
            {
                "name": "eth0",
                "mac": "4c:d5:87:4d:5e:81",
                "up": True,
                "mtu": 1500,
                "qlen": 1000
            }
        ]
    
    def _write_messages_log(self, logs: List[str]):
        """Write messages.log file"""
        filepath = self.output_dir / "messages.log"
        with open(filepath, 'w') as f:
            f.write('\n'.join(logs))
    
    def _write_showtech(self, showtech_sim: ShowTechSimulator, 
                       interfaces: Dict):
        """Write showtech.txt file"""
        filepath = self.output_dir / "showtech.txt"
        with open(filepath, 'w') as f:
            f.write(showtech_sim.generate_header())
            f.write(showtech_sim.generate_clock())
            f.write(showtech_sim.generate_version())
            f.write(showtech_sim.generate_interface_dom(interfaces))
            f.write(showtech_sim.generate_interface_brief(interfaces))
    
    def _write_routeinfo(self, route_sim: RouteInfoSimulator,
                        neighbors: List, routes: List, ifaces: List):
        """Write routeinfo.txt file"""
        filepath = self.output_dir / "routeinfo.txt"
        with open(filepath, 'w') as f:
            f.write(route_sim.generate_neighbor_show(neighbors))
            f.write(route_sim.generate_route_show(routes, "10.233.255.91"))
            f.write(route_sim.generate_interface_stats(ifaces))
    
    def _write_metadata(self, scenario: FailureScenario):
        """Write metadata.json with ground truth"""
        filepath = self.output_dir / "metadata.json"
        metadata = {
            "bundle_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "hostname": self.hostname,
            "ground_truth": scenario.get_ground_truth()
        }
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate enterprise network support bundles with injected failures"
    )
    parser.add_argument("--scenario", required=True,
                       choices=["port_flap", "bgp_neighbor_down", 
                               "vlan_mismatch", "missing_route"],
                       help="Failure scenario to inject")
    parser.add_argument("--output", required=True,
                       help="Output directory for support bundle")
    parser.add_argument("--duration", type=int, default=60,
                       help="Log duration in minutes (default: 60)")
    parser.add_argument("--noise-level", choices=["low", "medium", "high"],
                       default="medium",
                       help="Background noise level (default: medium)")
    parser.add_argument("--hostname", default="PSCSCTLEAFB03",
                       help="Switch hostname (default: PSCSCTLEAFB03)")
    
    # Scenario-specific arguments
    parser.add_argument("--interface", help="Interface name (for port_flap, vlan_mismatch)")
    parser.add_argument("--neighbor-ip", help="BGP neighbor IP (for bgp_neighbor_down)")
    parser.add_argument("--destination", help="Destination network (for missing_route)")
    parser.add_argument("--expected-vlan", type=int, help="Expected VLAN ID (for vlan_mismatch)")
    parser.add_argument("--received-vlan", type=int, help="Received VLAN ID (for vlan_mismatch)")
    
    args = parser.parse_args()
    
    # Build scenario kwargs
    scenario_kwargs = {}
    if args.interface:
        scenario_kwargs["interface"] = args.interface
    if args.neighbor_ip:
        scenario_kwargs["neighbor_ip"] = args.neighbor_ip
    if args.destination:
        scenario_kwargs["destination"] = args.destination
    if args.expected_vlan:
        scenario_kwargs["expected_vlan"] = args.expected_vlan
    if args.received_vlan:
        scenario_kwargs["received_vlan"] = args.received_vlan
    
    # Generate bundle
    generator = SupportBundleGenerator(args.output, args.hostname)
    generator.generate(
        scenario_name=args.scenario,
        duration_minutes=args.duration,
        noise_level=args.noise_level,
        **scenario_kwargs
    )


if __name__ == "__main__":
    main()
