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
            
            # Simple approach: use flows that encountered congested baselines as denominator
            flows_with_congested_baseline = len(getattr(self.lb, 'flows_with_congested_baseline', set()))
            
            stats['congestion_avoidance_percentage'] = stats.get('congestion_avoidance_rate', 0)
            stats['total_congestion_avoidance_events'] = total_events
            stats['unique_flows_with_congestion_avoidance'] = unique_flows
            stats['flows_with_congested_baseline'] = flows_with_congested_baseline
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
            
            # Add enhanced load distribution metrics if available
            if hasattr(self.lb, 'efficiency_tracker') and hasattr(self.lb.efficiency_tracker, 'load_distribution_metrics'):
                load_dist_metrics = self.lb.efficiency_tracker.load_distribution_metrics
                stats['load_distribution'] = {
                    'coefficient_of_variation': load_dist_metrics.get('coefficient_of_variation', 0),
                    'distribution_entropy': load_dist_metrics.get('distribution_entropy', 0),
                    'utilization_std_dev': load_dist_metrics.get('utilization_std_dev', 0),
                    'load_balancing_effectiveness': load_dist_metrics.get('load_balancing_effectiveness', 0),
                    'variance_reduction': load_dist_metrics.get('variance_reduction', 0),
                    'utilization_balance_score': load_dist_metrics.get('utilization_balance_score', 0),
                    'time_weighted_lb_rate': load_dist_metrics.get('time_weighted_lb_rate', 0),
                    'avg_utilization': load_dist_metrics.get('avg_utilization', 0),
                    'max_utilization': load_dist_metrics.get('max_utilization', 0),
                    'min_utilization': load_dist_metrics.get('min_utilization', 0),
                    'utilization_range': load_dist_metrics.get('utilization_range', 0)
                }
                
                # Override legacy load_balancing_rate with traffic-based calculation
                stats['load_balancing_rate'] = load_dist_metrics.get('load_balancing_effectiveness', 0)
                stats['legacy_load_balancing_rate'] = self.lb.efficiency_metrics.get('load_balancing_rate', 0)
            else:
                # Fallback to empty metrics
                stats['load_distribution'] = {
                    'coefficient_of_variation': 0,
                    'distribution_entropy': 0,
                    'utilization_std_dev': 0,
                    'load_balancing_effectiveness': 0,
                    'variance_reduction': 0,
                    'utilization_balance_score': 0,
                    'time_weighted_lb_rate': 0,
                    'avg_utilization': 0,
                    'max_utilization': 0,
                    'min_utilization': 0,
                    'utilization_range': 0
                }
        
        # Add additional event details for dashboard (percentage already calculated correctly by efficiency tracker)
        total_events = getattr(self.lb, 'total_congestion_avoidance_events', 0)
        unique_flows = len(getattr(self.lb, 'flows_with_congestion_avoidance', set()))
        flows_with_congested_baseline = len(getattr(self.lb, 'flows_with_congested_baseline', set()))
        
        # Add enhanced path-based congestion avoidance metrics if available
        if hasattr(self.lb, 'efficiency_tracker') and hasattr(self.lb.efficiency_tracker, 'path_calculator'):
            import time
            now = time.time()
            path_calc = self.lb.efficiency_tracker.path_calculator
            
            # Enhanced calculations
            stats['enhanced_path_congestion_avoidance'] = path_calc.calculate_path_congestion_avoidance(now)
            stats['enhanced_weighted_path_congestion_avoidance'] = path_calc.calculate_weighted_path_congestion_avoidance(now)
            stats['enhanced_binary_path_congestion_avoidance'] = path_calc.calculate_binary_path_state(now)
            
            # Configuration information for dashboard
            stats['congestion_calculation_config'] = path_calc.get_enhanced_configuration()
            
            # Algorithm metadata
            stats['algorithm_version'] = '2.0-enhanced'
            stats['algorithm_features'] = [
                'gradient_based_scoring',
                'temporal_trend_analysis', 
                'application_aware_prioritization',
                'multi_objective_optimization',
                'adaptive_thresholds'
            ]
        
        # Use the correctly calculated time-based percentage from efficiency tracker
        stats['congestion_avoidance_percentage'] = stats.get('congestion_avoidance_rate', 0)
        stats['total_congestion_avoidance_events'] = total_events
        stats['unique_flows_with_congestion_avoidance'] = unique_flows
        stats['flows_with_congested_baseline'] = flows_with_congested_baseline
        
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
    
    @route('load_distribution', '/stats/load_distribution', methods=['GET'])
    def load_distribution_stats(self, req, **_):
        """Get detailed load distribution metrics"""
        try:
            if hasattr(self.lb, 'efficiency_tracker') and hasattr(self.lb.efficiency_tracker, 'load_distribution_calculator'):
                calc = self.lb.efficiency_tracker.load_distribution_calculator
                
                # Get comprehensive load distribution summary
                summary = calc.get_network_load_summary()
                
                # Add calculation methodology information
                summary['methodology'] = {
                    'calculation_type': 'traffic_distribution_based',
                    'time_window_sec': calc.window_size_sec,
                    'min_samples_required': calc.min_samples,
                    'metrics_explanations': {
                        'coefficient_of_variation': 'Lower values indicate better load balancing (std_dev/mean)',
                        'distribution_entropy': 'Higher values indicate better load balancing (0-1 normalized)',
                        'load_balancing_effectiveness': 'Overall effectiveness percentage (0-100%)',
                        'utilization_balance_score': 'How evenly traffic is distributed (0-100%)',
                        'variance_reduction': 'Improvement over shortest-path routing (%)'
                    }
                }
                
                return self._cors(json.dumps(summary))
            else:
                return self._cors(json.dumps({
                    'error': 'Load distribution calculator not available',
                    'load_distribution': {
                        'coefficient_of_variation': 0,
                        'distribution_entropy': 0,
                        'load_balancing_effectiveness': 0,
                        'utilization_balance_score': 0,
                        'variance_reduction': 0
                    }
                }))
                
        except Exception as e:
            self.lb.logger.error("Error getting load distribution stats: %s", e)
            return self._cors(json.dumps({'error': str(e)}), 500)
    
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
    
    @route('enhanced_config', '/config/enhanced', methods=['GET'])
    def get_enhanced_config(self, req, **_):
        """Get enhanced congestion avoidance calculation configuration"""
        if (hasattr(self.lb, 'efficiency_tracker') and 
            hasattr(self.lb.efficiency_tracker, 'path_calculator')):
            config = self.lb.efficiency_tracker.path_calculator.get_enhanced_configuration()
            return self._cors(json.dumps(config))
        else:
            return self._cors(json.dumps({'error': 'Enhanced calculator not available'}), 404)
    
    @route('enhanced_config_update', '/config/enhanced', methods=['POST'])
    def update_enhanced_config(self, req, **_):
        """Update enhanced congestion avoidance calculation weights"""
        try:
            if not (hasattr(self.lb, 'efficiency_tracker') and 
                   hasattr(self.lb.efficiency_tracker, 'path_calculator')):
                return self._cors(json.dumps({'error': 'Enhanced calculator not available'}), 404)
            
            data = json.loads(req.body.decode('utf-8'))
            path_calc = self.lb.efficiency_tracker.path_calculator
            
            # Update configuration weights if provided
            if 'utilization_weight' in data:
                path_calc.config['utilization_weight'] = float(data['utilization_weight'])
            if 'trend_weight' in data:
                path_calc.config['trend_weight'] = float(data['trend_weight'])
            if 'capacity_weight' in data:
                path_calc.config['capacity_weight'] = float(data['capacity_weight'])
            if 'latency_weight' in data:
                path_calc.config['latency_weight'] = float(data['latency_weight'])
            if 'reliability_weight' in data:
                path_calc.config['reliability_weight'] = float(data['reliability_weight'])
            
            # Validate weights sum to approximately 1.0
            total_weight = (path_calc.config['utilization_weight'] + 
                          path_calc.config['trend_weight'] + 
                          path_calc.config['capacity_weight'] + 
                          path_calc.config['latency_weight'] + 
                          path_calc.config['reliability_weight'])
            
            if abs(total_weight - 1.0) > 0.1:
                return self._cors(json.dumps({'error': f'Weights must sum to approximately 1.0, got {total_weight}'}), 400)
            
            self.lb.logger.info("Enhanced calculation weights updated: util=%.2f, trend=%.2f, capacity=%.2f, latency=%.2f, reliability=%.2f", 
                              path_calc.config['utilization_weight'], path_calc.config['trend_weight'],
                              path_calc.config['capacity_weight'], path_calc.config['latency_weight'], 
                              path_calc.config['reliability_weight'])
            
            return self._cors(json.dumps({'success': True, 'new_config': path_calc.get_enhanced_configuration()}))
        
        except Exception as e:
            return self._cors(json.dumps({'error': str(e)}), 400)
    
    @route('options', '/{path:.*}', methods=['OPTIONS'])
    def options_handler(self, req, **_):
        """Handle preflight OPTIONS requests"""
        return self._cors('', 200)