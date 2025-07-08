"""
REST API Server
===============

Provides HTTP REST API endpoints for monitoring and configuring
the SDN load balancer.
"""

import json
import time
from ryu.app.wsgi import ControllerBase, route, Response


class LBRestController(ControllerBase):
    """REST API controller for load balancer monitoring and configuration"""
    
    def __init__(self, req, link, data, **cfg):
        super().__init__(req, link, data, **cfg)
        self.lb = data['lbapp']
    
    def _cors(self, body, status=200):
        """Add CORS headers to response"""
        return Response(
            body=body, status=status,
            content_type='application/json',
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
    
    @route('path', '/load/path', methods=['GET'])
    def get_paths(self, req, **_):
        """Get current flow paths"""
        paths = {}
        for (src, dst), path in self.lb.flow_paths.items():
            src_name = self.lb.hosts.get(src, src)
            dst_name = self.lb.hosts.get(dst, dst)
            label = f"{src_name}→{dst_name}"
            paths[label] = path
        return self._cors(json.dumps(paths))
    
    @route('links', '/load/links', methods=['GET'])
    def get_links(self, req, **_):
        """Get current link utilization"""
        now = time.time()
        data = {}
        
        # Use traffic monitor if available
        if hasattr(self.lb, 'traffic_monitor'):
            data = self.lb.traffic_monitor.get_all_link_loads(now)
        else:
            # Fallback to direct calculation
            for (u, v), (pu, pv) in self.lb.links.items():
                if u < v:  # Avoid duplicates
                    key = f"{u}-{v}"
                    rate_u = self.lb._avg_rate(u, pu, now)
                    rate_v = self.lb._avg_rate(v, pv, now)
                    data[key] = max(rate_u, rate_v)
        
        return self._cors(json.dumps(data))
    
    @route('topology', '/topology', methods=['GET'])
    def get_topology(self, req, **_):
        """Return current network topology for dynamic visualization"""
        nodes = []
        links = []
        
        # Validate host consistency
        if hasattr(self.lb, 'host_manager'):
            consistency_issues = self.lb.host_manager.validate_host_consistency()
            if consistency_issues:
                self.lb.logger.warning("Host consistency issues detected: %s", consistency_issues)
        
        # Debug logging
        total_hosts_in_hosts_dict = len(self.lb.hosts)
        total_hosts_in_locations = sum(len(macs) for macs in self.lb.host_locations.values())
        
        self.lb.logger.debug("Topology API: %d hosts in dict, %d hosts in locations", 
                           total_hosts_in_hosts_dict, total_hosts_in_locations)
        
        # Add switch nodes with host count information
        for dpid in self.lb.dp_set.keys():
            host_count = len(self.lb.host_locations.get(dpid, set()))
            nodes.append({
                "id": f"s{dpid}", 
                "type": "switch", 
                "host_count": host_count
            })
        
        # Add host nodes
        valid_hosts = {}
        added_host_names = set()
        
        for mac, host_name in self.lb.hosts.items():
            if mac in self.lb.mac_to_dpid:
                dpid = self.lb.mac_to_dpid[mac]
                
                # Show host even if switch temporarily unavailable
                if dpid in self.lb.dp_set or True:
                    # Only add if we haven't seen this host name before
                    if host_name not in added_host_names:
                        nodes.append({"id": host_name, "type": "host", "mac": mac})
                        added_host_names.add(host_name)
                    valid_hosts[mac] = (host_name, dpid)
                    
                    # Log missing switch but don't hide host
                    if dpid not in self.lb.dp_set:
                        self.lb.logger.debug("Host %s (MAC: %s) references switch %d not in dp_set", 
                                           host_name, mac, dpid)
            else:
                self.lb.logger.debug("Host %s (MAC: %s) not found in mac_to_dpid", 
                                   host_name, mac)
        
        # Add switch-to-switch links
        added_links = set()
        for (u, v) in self.lb.links.keys():
            if u < v and (u, v) not in added_links:
                links.append({"source": f"s{u}", "target": f"s{v}", "type": "switch-switch"})
                added_links.add((u, v))
        
        # Add host-to-switch links
        for mac, (host_name, dpid) in valid_hosts.items():
            if dpid in self.lb.dp_set:
                links.append({"source": host_name, "target": f"s{dpid}", "type": "host-switch"})
        
        topology = {"nodes": nodes, "links": links}
        return self._cors(json.dumps(topology))
    
    @route('efficiency', '/stats/efficiency', methods=['GET'])
    def get_efficiency_stats(self, req, **_):
        """Get efficiency statistics"""
        if hasattr(self.lb, 'efficiency_tracker'):
            stats = self.lb.efficiency_tracker.get_efficiency_summary()
            
            # Add congestion avoidance percentage and event details for dashboard
            total_events = getattr(self.lb, 'total_congestion_avoidance_events', 0)
            unique_flows = len(getattr(self.lb, 'flows_with_congestion_avoidance', set()))
            
            stats['congestion_avoidance_percentage'] = stats.get('congestion_avoidance_rate', 0)
            stats['total_congestion_avoidance_events'] = total_events
            stats['unique_flows_with_congestion_avoidance'] = unique_flows
        else:
            # Fallback to direct calculation
            now = time.time()
            if hasattr(self.lb, '_calculate_efficiency_metrics'):
                self.lb._calculate_efficiency_metrics(now)
            
            stats = {
                'total_flows': self.lb.efficiency_metrics.get('total_flows', 0),
                'load_balanced_flows': self.lb.efficiency_metrics.get('load_balanced_flows', 0),
                'congestion_avoided': self.lb.efficiency_metrics.get('congestion_avoided', 0),
                'load_balancing_rate': self.lb.efficiency_metrics.get('load_balancing_rate', 0),
                'congestion_avoidance_rate': self.lb.efficiency_metrics.get('congestion_avoidance_rate', 0),
                'variance_improvement_percent': self.lb.efficiency_metrics.get('variance_improvement_percent', 0),
                'path_overhead_percent': self.lb.efficiency_metrics.get('path_overhead_percent', 0),
                'total_reroutes': self.lb.efficiency_metrics.get('total_reroutes', 0),
                'runtime_minutes': self.lb.efficiency_metrics.get('runtime_minutes', 0),
                'avg_path_length_lb': self.lb.efficiency_metrics.get('avg_path_length_lb', 0),
                'avg_path_length_sp': self.lb.efficiency_metrics.get('avg_path_length_sp', 0)
            }
        
        # Add congestion avoidance percentage calculation for dashboard
        total_events = getattr(self.lb, 'total_congestion_avoidance_events', 0)
        unique_flows = len(getattr(self.lb, 'flows_with_congestion_avoidance', set()))
        
        stats['congestion_avoidance_percentage'] = stats['congestion_avoidance_rate']
        stats['total_congestion_avoidance_events'] = total_events
        stats['unique_flows_with_congestion_avoidance'] = unique_flows
        
        return self._cors(json.dumps(stats))
    
    @route('algorithm', '/stats/algorithm', methods=['GET'])
    def get_algorithm_stats(self, req, **_):
        """Get algorithm statistics"""
        # Map mode number to name
        from ..config.constants import LOAD_BALANCING_MODES
        mode_names = {v: k for k, v in LOAD_BALANCING_MODES.items()}
        
        current_mode = mode_names.get(self.lb.load_balancing_mode, 'unknown')
        
        algorithm_stats = {
            'alternative_paths_stored': len(getattr(self.lb, 'alternative_paths', {})),
            'congestion_trends_tracked': len(getattr(self.lb, 'congestion_trends', {})),
            'flow_classifications': len(getattr(self.lb, 'flow_characteristics', {})),
            'ecmp_flow_entries': len(getattr(self.lb, 'ecmp_flow_table', {}))
        }
        
        stats = {
            'current_mode': current_mode,
            'algorithm_stats': algorithm_stats
        }
        
        return self._cors(json.dumps(stats))
    
    @route('threshold_get', '/config/threshold', methods=['GET'])
    def get_threshold(self, req, **_):
        """Get current congestion threshold"""
        return self._cors(json.dumps({'threshold': self.lb.THRESHOLD_BPS}))
    
    @route('threshold_set', '/config/threshold', methods=['POST'])
    def set_threshold(self, req, **_):
        """Set congestion threshold"""
        try:
            data = json.loads(req.body.decode('utf-8'))
            new_threshold = data.get('threshold', self.lb.THRESHOLD_BPS)
            
            if new_threshold <= 0:
                return self._cors(json.dumps({'error': 'Threshold must be positive'}), 400)
            
            self.lb.THRESHOLD_BPS = new_threshold
            self.lb.logger.info("Threshold updated to %d bytes/sec (%.1f Mbps)", 
                              new_threshold, new_threshold / 1_000_000)
            
            return self._cors(json.dumps({'success': True, 'threshold': new_threshold}))
        
        except Exception as e:
            return self._cors(json.dumps({'error': str(e)}), 400)
    
    @route('mode_get', '/config/mode', methods=['GET'])
    def get_mode(self, req, **_):
        """Get current load balancing mode"""
        # Map mode number to name
        from ..config.constants import LOAD_BALANCING_MODES
        mode_names = {v: k for k, v in LOAD_BALANCING_MODES.items()}
        current_mode = mode_names.get(self.lb.load_balancing_mode, 'unknown')
        
        return self._cors(json.dumps({'mode': current_mode}))
    
    @route('mode_set', '/config/mode', methods=['POST'])
    def set_mode(self, req, **_):
        """Set load balancing mode"""
        try:
            data = json.loads(req.body.decode('utf-8'))
            new_mode = data.get('mode', '').lower()
            
            from ..config.constants import LOAD_BALANCING_MODES
            if new_mode not in LOAD_BALANCING_MODES:
                return self._cors(json.dumps({'error': f'Invalid mode: {new_mode}'}), 400)
            
            old_mode = self.lb.load_balancing_mode
            self.lb.load_balancing_mode = LOAD_BALANCING_MODES[new_mode]
            
            # Reset efficiency metrics when mode changes
            if hasattr(self.lb, 'efficiency_tracker'):
                self.lb.efficiency_tracker.reset_efficiency_metrics()
            else:
                # Fallback reset - count existing flows immediately
                existing_flows_count = len(getattr(self.lb, 'flow_paths', {}))
                self.lb.efficiency_metrics = {
                    'total_flows': existing_flows_count,  # Count existing flows immediately
                    'load_balanced_flows': 0,  # Reset - new mode will route differently
                    'congestion_avoided': 0,   # Reset - new mode may avoid congestion differently
                    'avg_path_length_lb': 0,
                    'avg_path_length_sp': 0,
                    'total_reroutes': 0,       # Reset - fresh count for new mode
                    'link_utilization_variance': 0,
                    'baseline_link_utilization_variance': 0,
                    'start_time': time.time()  # Reset runtime for new mode
                }
                
                # Clear congestion avoidance tracking for new mode
                if hasattr(self.lb, 'flows_with_congestion_avoidance'):
                    self.lb.flows_with_congestion_avoidance.clear()
                if hasattr(self.lb, 'congestion_avoidance_events'):
                    self.lb.congestion_avoidance_events.clear()
                if hasattr(self.lb, 'total_congestion_avoidance_events'):
                    self.lb.total_congestion_avoidance_events = 0
            
            self.lb.logger.info("Load balancing mode changed from %s to %s, efficiency metrics reset", 
                              old_mode, new_mode)
            
            return self._cors(json.dumps({'success': True, 'mode': new_mode}))
        
        except Exception as e:
            return self._cors(json.dumps({'error': str(e)}), 400)
    
    @route('debug_flows', '/debug/flows', methods=['GET'])
    def debug_flows(self, req, **_):
        """Debug endpoint for flow information"""
        if hasattr(self.lb, 'flow_classifier'):
            stats = self.lb.flow_classifier.get_flow_statistics()
        else:
            stats = {
                'total_flows': len(getattr(self.lb, 'flow_characteristics', {})),
                'flow_paths': len(self.lb.flow_paths)
            }
        
        return self._cors(json.dumps(stats))
    
    @route('debug_topology', '/debug/topology', methods=['GET'])
    def debug_topology(self, req, **_):
        """Debug endpoint for topology information"""
        if hasattr(self.lb, 'topology_manager'):
            info = self.lb.topology_manager.get_topology_info()
        else:
            info = {
                'switches': len(self.lb.dp_set),
                'links': len(self.lb.links),
                'hosts': len(self.lb.hosts),
                'ready': getattr(self.lb, 'topology_ready', False)
            }
        
        return self._cors(json.dumps(info))
    
    @route('debug_efficiency', '/debug/efficiency', methods=['GET'])
    def debug_efficiency(self, req, **_):
        """Debug endpoint for efficiency metrics"""
        if hasattr(self.lb, 'efficiency_tracker'):
            stats = self.lb.efficiency_tracker.get_detailed_stats()
        else:
            stats = {
                'metrics': getattr(self.lb, 'efficiency_metrics', {}),
                'flows_with_congestion_avoidance': len(getattr(self.lb, 'flows_with_congestion_avoidance', set()))
            }
        
        return self._cors(json.dumps(stats))
    
    @route('debug_congestion', '/debug/congestion', methods=['GET'])
    def debug_congestion(self, req, **_):
        """Debug endpoint for congestion prediction"""
        if hasattr(self.lb, 'congestion_predictor'):
            stats = self.lb.congestion_predictor.get_prediction_statistics()
        else:
            stats = {
                'congestion_trends': len(getattr(self.lb, 'congestion_trends', {})),
                'congestion_ewma': len(getattr(self.lb, 'congestion_ewma', {}))
            }
        
        return self._cors(json.dumps(stats))
    
    @route('debug_traffic', '/debug/traffic', methods=['GET'])
    def debug_traffic(self, req, **_):
        """Debug endpoint for traffic statistics"""
        if hasattr(self.lb, 'traffic_monitor'):
            stats = self.lb.traffic_monitor.get_network_statistics()
        else:
            stats = {
                'total_switches': len(self.lb.dp_set),
                'total_links': len(self.lb.links) // 2,
                'rate_history_entries': sum(len(ports) for ports in getattr(self.lb, 'rate_hist', {}).values())
            }
        
        return self._cors(json.dumps(stats))
    
    @route('options', '/{path:.*}', methods=['OPTIONS'])
    def options_handler(self, req, **_):
        """Handle preflight OPTIONS requests"""
        return self._cors('', 200)