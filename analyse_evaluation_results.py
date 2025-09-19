#!/usr/bin/env python3
"""
Analyse evaluation results from intermediate JSON files.
Creates plots and computes average metrics.
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import re

class EvaluationResultsAnalyzer:
    """Analyse and visualise evaluation results from intermediate files."""
    
    def __init__(self, results_dir: str = "beta_evaluation_10k_results"):
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.metrics_over_time = {
            'samples': [],
            'inception_score': [],
            'fid_score': [],
            'constraint_satisfaction': [],
            'success_rate': []
        }
        
    def find_intermediate_files(self) -> List[Path]:
        """Find all intermediate results files."""
        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            return []
        
        # Find all intermediate_results_*.json files
        pattern = str(self.results_dir / "intermediate_results_*.json")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No intermediate results files found in {self.results_dir}")
            return []
        
        # Sort by sample count
        def extract_sample_count(filename):
            match = re.search(r'intermediate_results_(\d+)\.json', filename)
            return int(match.group(1)) if match else 0
        
        sorted_files = sorted(files, key=extract_sample_count)
        print(f"Found {len(sorted_files)} intermediate results files")
        
        return [Path(f) for f in sorted_files]
    
    def load_results(self) -> bool:
        """Load all intermediate results files."""
        files = self.find_intermediate_files()
        if not files:
            return False
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract sample count from filename
                sample_count = int(re.search(r'intermediate_results_(\d+)\.json', file_path.name).group(1))
                
                # Store the data with sample count
                result_entry = {
                    'sample_count': sample_count,
                    'file_path': str(file_path),
                    'data': data
                }
                self.results_data.append(result_entry)
                
                # Calculate metrics from individual results (no top-level metrics exist)
                calculated_metrics = self._calculate_metrics_from_results(data)
                
                self.metrics_over_time['samples'].append(sample_count)
                self.metrics_over_time['inception_score'].append(calculated_metrics['inception_score'])
                self.metrics_over_time['fid_score'].append(calculated_metrics['fid_score'])
                self.metrics_over_time['constraint_satisfaction'].append(calculated_metrics['constraint_satisfaction'])
                
                success_rate = data.get('success_rate', 0)
                self.metrics_over_time['success_rate'].append(success_rate)
                
                print(f"Loaded {file_path.name}: {sample_count} samples, IS={calculated_metrics['inception_score']:.3f}, "
                      f"FID={calculated_metrics['fid_score']:.1f}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.results_data)} result files")
        return len(self.results_data) > 0
    
    def _calculate_metrics_from_results(self, data: Dict) -> Dict[str, float]:
        """Calculate aggregated metrics from individual results with mathematical rigor."""
        results = data.get('results', [])
        if not results:
            return {
                'inception_score': 0.0,
                'fid_score': 0.0,
                'constraint_satisfaction': 0.0
            }
        
        # Filter successful results only for metric calculations
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'inception_score': 0.0,
                'fid_score': 0.0,
                'constraint_satisfaction': 0.0
            }
        
        # Calculate weighted averages for each metric
        inception_scores = [r.get('is', 0.0) for r in successful_results if 'is' in r]
        fid_scores = [r.get('fid', 0.0) for r in successful_results if 'fid' in r]
        
        # Calculate constraint satisfaction with mathematical precision
        constraint_satisfactions = []
        for result in successful_results:
            if 'constraint_satisfaction' in result:
                sat_rate = result['constraint_satisfaction'].get('satisfaction_rate', 0.0)
                constraint_satisfactions.append(sat_rate)
        
        # Calculate averages with proper handling of empty lists
        avg_is = np.mean(inception_scores) if inception_scores else 0.0
        avg_fid = np.mean(fid_scores) if fid_scores else 0.0
        avg_constraint = np.mean(constraint_satisfactions) if constraint_satisfactions else 0.0
        
        return {
            'inception_score': float(avg_is),
            'fid_score': float(avg_fid),
            'constraint_satisfaction': float(avg_constraint)
        }
    
    def _calculate_avg_constraint_satisfaction(self, data: Dict) -> float:
        """Calculate average constraint satisfaction from results."""
        results = data.get('results', [])
        if not results:
            return 0.0
        
        constraint_satisfactions = []
        for result in results:
            if result.get('success', False) and 'constraint_satisfaction' in result:
                satisfaction_rate = result['constraint_satisfaction'].get('satisfaction_rate', 0.0)
                constraint_satisfactions.append(satisfaction_rate)
        
        return float(np.mean(constraint_satisfactions)) if constraint_satisfactions else 0.0
    
    def calculate_overall_averages(self) -> Dict[str, float]:
        """Calculate overall average metrics across all results."""
        if not self.results_data:
            return {}
        
        # Use the latest (most comprehensive) results file for overall averages
        latest_data = max(self.results_data, key=lambda x: x['sample_count'])['data']
        
        # Calculate metrics from individual results (no top-level metrics exist)
        calculated_metrics = self._calculate_metrics_from_results(latest_data)
        
        averages = {
            'total_samples': latest_data.get('processed_samples', 0),
            'success_rate': latest_data.get('success_rate', 0.0),
            'inception_score': calculated_metrics['inception_score'],
            'fid_score': calculated_metrics['fid_score'],
            'constraint_satisfaction': calculated_metrics['constraint_satisfaction']
        }
        
        return averages
    
    def _calculate_room_type_stats(self, data: Dict) -> Dict[str, Dict]:
        """Calculate room type statistics from individual results."""
        results = data.get('results', [])
        if not results:
            return {}
        
        room_stats = {}
        
        for result in results:
            if not result.get('success', False):
                continue
                
            room_type = result.get('room_type', 'unknown')
            
            if room_type not in room_stats:
                room_stats[room_type] = {
                    'count': 0,
                    'inception_scores': [],
                    'fid_scores': [],
                    'constraint_satisfactions': []
                }
            
            room_stats[room_type]['count'] += 1
            
            # Collect metrics for this room type
            if 'is' in result:
                room_stats[room_type]['inception_scores'].append(result['is'])
            if 'fid' in result:
                room_stats[room_type]['fid_scores'].append(result['fid'])
            if 'constraint_satisfaction' in result:
                sat_rate = result['constraint_satisfaction'].get('satisfaction_rate', 0.0)
                room_stats[room_type]['constraint_satisfactions'].append(sat_rate)
        
        # Calculate averages for each room type
        final_room_stats = {}
        for room_type, stats in room_stats.items():
            final_room_stats[room_type] = {
                'count': stats['count'],
                'avg_inception_score': float(np.mean(stats['inception_scores'])) if stats['inception_scores'] else 0.0,
                'avg_fid_score': float(np.mean(stats['fid_scores'])) if stats['fid_scores'] else 0.0,
                'avg_constraint_satisfaction': float(np.mean(stats['constraint_satisfactions'])) if stats['constraint_satisfactions'] else 0.0
            }
        
        return final_room_stats
    
    def _setup_plot_style(self):
        """Set up consistent plotting style."""
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 12
        })
    
    def _create_results_directory(self):
        """Create results directory if it doesn't exist."""
        results_path = Path('results')
        results_path.mkdir(exist_ok=True)
        return results_path
    
    def _plot_inception_score_over_time(self, results_path):
        """Create Inception Score over time plot."""
        self._setup_plot_style()
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_over_time['samples'], self.metrics_over_time['inception_score'], 
                'o-', linewidth=3, markersize=8, color='blue')
        plt.title('Inception Score Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Samples', fontsize=14)
        plt.ylabel('Inception Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = results_path / '01_inception_score_over_time.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    
    def _plot_fid_score_over_time(self, results_path):
        """Create FID Score over time plot."""
        self._setup_plot_style()
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_over_time['samples'], self.metrics_over_time['fid_score'], 
                'o-', linewidth=3, markersize=8, color='orange')
        plt.title('FID Score Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Samples', fontsize=14)
        plt.ylabel('FID Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = results_path / '02_fid_score_over_time.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    
    def _plot_clip_score_over_time(self, results_path):
        """Create CLIP Score over time plot."""
        self._setup_plot_style()
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_over_time['samples'], self.metrics_over_time['clip_score'], 
                'o-', linewidth=3, markersize=8, color='green')
        plt.title('CLIP Score Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Samples', fontsize=14)
        plt.ylabel('CLIP Score', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = results_path / '03_clip_score_over_time.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    
    def _plot_constraint_satisfaction_and_success_rate(self, results_path):
        """Create constraint satisfaction and success rate plot."""
        self._setup_plot_style()
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics_over_time['samples'], 
                [cs * 100 for cs in self.metrics_over_time['constraint_satisfaction']], 
                'o-', color='purple', linewidth=3, markersize=8, label='Constraint Satisfaction')
        plt.plot(self.metrics_over_time['samples'], 
                [sr * 100 for sr in self.metrics_over_time['success_rate']], 
                'o-', color='red', linewidth=3, markersize=8, label='Success Rate')
        plt.title('Constraint Satisfaction & Success Rate', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Samples', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = results_path / '04_constraint_satisfaction_success_rate.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    
    def _plot_normalized_metrics_comparison(self, results_path):
        """Create normalized metrics comparison plot."""
        self._setup_plot_style()
        plt.figure(figsize=(12, 6))
        
        # Normalize metrics for comparison (0-1 scale)
        def normalize_metric(values, target_range=(0, 1)):
            values = np.array(values)
            if len(values) == 0 or np.max(values) == np.min(values):
                return values
            normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
            return normalized * (target_range[1] - target_range[0]) + target_range[0]
        
        norm_is = normalize_metric(self.metrics_over_time['inception_score'])
        norm_fid = 1 - normalize_metric(self.metrics_over_time['fid_score'])  # Invert FID (lower is better)
        norm_constraint = self.metrics_over_time['constraint_satisfaction']
        
        plt.plot(self.metrics_over_time['samples'], norm_is, 'o-',
                label='Inception Score (norm)', linewidth=3, markersize=8)
        plt.plot(self.metrics_over_time['samples'], norm_fid, 'o-',
                label='FID Score (norm, inverted)', linewidth=3, markersize=8)
        plt.plot(self.metrics_over_time['samples'], norm_constraint, 'o-',
                label='Constraint Satisfaction', linewidth=3, markersize=8)
        
        plt.title('Normalized Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Number of Samples', fontsize=14)
        plt.ylabel('Normalized Score (0-1)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = results_path / '05_normalized_metrics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    
    def _plot_final_metrics_values(self, results_path):
        """Create final metrics values bar chart."""
        if len(self.metrics_over_time['samples']) <= 1:
            return None
            
        self._setup_plot_style()
        plt.figure(figsize=(10, 6))
        
        metrics_for_hist = [
            self.metrics_over_time['inception_score'][-1],
            self.metrics_over_time['constraint_satisfaction'][-1],
            self.metrics_over_time['success_rate'][-1]
        ]
        labels = ['Inception\nScore', 'Constraint\nSatisfaction', 'Success\nRate']
        colors = ['blue', 'purple', 'red']
        
        bars = plt.bar(labels, metrics_for_hist, color=colors, alpha=0.8)
        plt.title('Final Metrics Values', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Score', fontsize=14)
        plt.ylim(0, max(metrics_for_hist) * 1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_for_hist):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        
        output_path = results_path / '06_final_metrics_values.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    
    def _plot_room_type_analysis(self, results_path):
        """Create room type constraint satisfaction plot."""
        self._setup_plot_style()
        plt.figure(figsize=(12, 6))
        
        latest_data = max(self.results_data, key=lambda x: x['sample_count'])['data']
        room_stats = self._calculate_room_type_stats(latest_data)
        
        if room_stats:
            room_types = list(room_stats.keys())
            constraint_sats = [room_stats[room]['avg_constraint_satisfaction'] * 100 
                             for room in room_types]
            
            bars = plt.bar(room_types, constraint_sats, color='skyblue', alpha=0.8)
            plt.title('Constraint Satisfaction by Room Type', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Constraint Satisfaction (%)', fontsize=14)
            plt.xlabel('Room Type', fontsize=14)
            plt.xticks(rotation=45)
            
            for bar, value in zip(bars, constraint_sats):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        else:
            plt.text(0.5, 0.5, 'No room type data available', ha='center', va='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.title('Room Type Analysis', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_path = results_path / '07_room_type_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    
    def _plot_processing_efficiency(self, results_path):
        """Create processing efficiency plot."""
        self._setup_plot_style()
        plt.figure(figsize=(8, 6))
        
        if len(self.metrics_over_time['samples']) > 1:
            # Calculate samples per time unit
            sample_increments = np.diff(self.metrics_over_time['samples'])
            avg_increment = np.mean(sample_increments)
            
            plt.bar(['Average Samples\nPer Batch'], [avg_increment], color='lightcoral', alpha=0.8, width=0.5)
            plt.title('Processing Efficiency', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Samples per Batch', fontsize=14)
            
            plt.text(0, avg_increment + avg_increment*0.1, f'{avg_increment:.0f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=14)
        else:
            plt.bar(['Single Batch'], [self.metrics_over_time['samples'][0]], 
                   color='lightcoral', alpha=0.8, width=0.5)
            plt.title('Processing Efficiency', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Total Samples', fontsize=14)
            
            plt.text(0, self.metrics_over_time['samples'][0] + self.metrics_over_time['samples'][0]*0.1, 
                    f'{self.metrics_over_time["samples"][0]}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        output_path = results_path / '08_processing_efficiency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    
    def _create_summary_report(self, results_path):
        """Create text-based summary report."""
        averages = self.calculate_overall_averages()
        
        summary_text = f"""BETA DISTRIBUTION SPATIAL REASONER - EVALUATION SUMMARY
{'='*70}

DATA OVERVIEW:
  - Result files analyzed: {len(self.results_data)}
  - Total samples processed: {averages.get('total_samples', 0):,}
  - Overall success rate: {averages.get('success_rate', 0)*100:.1f}%

AVERAGE METRICS:
  - Inception Score: {averages.get('inception_score', 0):.4f}
  - FID Score: {averages.get('fid_score', 0):.2f}
  - Constraint Satisfaction: {averages.get('constraint_satisfaction', 0)*100:.1f}%

QUALITY ASSESSMENT:
  - IS Quality: {'Good' if averages.get('inception_score', 0) > 3.0 else 'Fair' if averages.get('inception_score', 0) > 2.0 else 'Needs Improvement'}
  - FID Quality: {'Good' if averages.get('fid_score', 0) < 100 else 'Fair' if averages.get('fid_score', 0) < 200 else 'Needs Improvement'}
  - Constraint Quality: {'Excellent' if averages.get('constraint_satisfaction', 0) > 0.8 else 'Good' if averages.get('constraint_satisfaction', 0) > 0.6 else 'Needs Improvement'}

GENERATED: {Path().cwd()}
{'='*70}
        """
        
        output_path = results_path / '00_evaluation_summary.txt'
        with open(output_path, 'w') as f:
            f.write(summary_text)
        
        return output_path
    
    def create_plots(self):
        """Create separate analysis plots and save to /results directory."""
        if not self.results_data:
            print("No data to plot")
            return
        
        # Create results directory
        results_path = self._create_results_directory()
        print(f"Creating individual plots in: {results_path}")
        
        created_files = []
        
        # Create summary report
        summary_path = self._create_summary_report(results_path)
        created_files.append(summary_path)
        
        # Create individual plots
        plots = [
            self._plot_inception_score_over_time(results_path),
            self._plot_fid_score_over_time(results_path),
            self._plot_constraint_satisfaction_and_success_rate(results_path),
            self._plot_normalized_metrics_comparison(results_path),
            self._plot_final_metrics_values(results_path),
            self._plot_room_type_analysis(results_path),
            self._plot_processing_efficiency(results_path)
        ]
        
        # Filter out None values and add to created files
        created_files.extend([p for p in plots if p is not None])
        
        print(f"\nCreated {len(created_files)} analysis files:")
        for file_path in created_files:
            print(f"  - {file_path.name}")
        
        print(f"\nAll analysis files saved in: {results_path.absolute()}")
        return created_files
    
    def print_summary(self):
        """Print a comprehensive summary of the evaluation results."""
        if not self.results_data:
            print("No data to summarize")
            return
        
        averages = self.calculate_overall_averages()
        
        print("\n" + "="*80)
        print("BETA DISTRIBUTION SPATIAL REASONING EVALUATION ANALYSIS")
        print("="*80)
        
        print(f"\nDATA OVERVIEW:")
        print(f"   • Result files analyzed: {len(self.results_data)}")
        print(f"   • Total samples processed: {averages.get('total_samples', 0):,}")
        print(f"   • Overall success rate: {averages.get('success_rate', 0)*100:.1f}%")
        
        print(f"\nAVERAGE METRICS:")
        print(f"   • Inception Score: {averages.get('inception_score', 0):.4f}")
        print(f"   • FID Score: {averages.get('fid_score', 0):.2f}")
        print(f"   • Constraint Satisfaction: {averages.get('constraint_satisfaction', 0)*100:.1f}%")
        
        print(f"\nMETRICS PROGRESSION:")
        if len(self.metrics_over_time['samples']) > 1:
            initial_is = self.metrics_over_time['inception_score'][0]
            final_is = self.metrics_over_time['inception_score'][-1]

            print(f"   • IS: {initial_is:.3f} → {final_is:.3f} (Δ: {final_is-initial_is:+.3f})")
        else:
            print("   • Single data point - no progression to analyze")
        
        print(f"\nQUALITY ASSESSMENT:")
        is_score = averages.get('inception_score', 0)
        fid_score = averages.get('fid_score', 0)
        constraint_sat = averages.get('constraint_satisfaction', 0)

        # Quality assessment based on typical benchmarks
        is_quality = "Good" if is_score > 3.0 else "Fair" if is_score > 2.0 else "Needs Improvement"
        fid_quality = "Good" if fid_score < 100 else "Fair" if fid_score < 200 else "Needs Improvement"
        constraint_quality = "Excellent" if constraint_sat > 0.8 else "Good" if constraint_sat > 0.6 else "Needs Improvement"

        print(f"   • Inception Score: {is_quality}")
        print(f"   • FID Score: {fid_quality}")
        print(f"   • Constraint Satisfaction: {constraint_quality}")
        
        # Room type breakdown
        latest_data = max(self.results_data, key=lambda x: x['sample_count'])['data']
        room_stats = self._calculate_room_type_stats(latest_data)
        
        if room_stats:
            print(f"\nROOM TYPE BREAKDOWN:")
            for room_type, stats in room_stats.items():
                print(f"   • {room_type.capitalize()}: {stats['count']} samples, "
                      f"CS={stats['avg_constraint_satisfaction']*100:.1f}%, "
                      f"IS={stats['avg_inception_score']:.3f}, "
                      f"FID={stats['avg_fid_score']:.1f}")
        
        print("\n" + "="*80)

def main():
    """Main function to run the analysis."""
    print("Starting evaluation results analysis...")
    
    analyzer = EvaluationResultsAnalyzer()
    
    if not analyzer.load_results():
        print("No results to analyze. Run some evaluations first.")
        return
    
    # Print comprehensive summary
    analyzer.print_summary()
    
    # Create and display plots
    print("\nCreating analysis plots...")
    analyzer.create_plots()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()