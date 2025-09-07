#!/usr/bin/env python3
"""
Batch processing script for analyzing multiple BRACS slides with annotations.
This script processes all slides sequentially and generates comprehensive reports.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import traceback
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import wandb

# Add your project path
sys.path.append('/vol/research/scratch1/NOBACKUP/rk01337/ACMIL')

# Import your heatmap generator
from generate_heatmap_all_models_bracs import BRACSHeatmapGenerator

class BatchBRACSAnalyzer:
    """
    Batch processor for multiple BRACS slides with comprehensive analysis
    """
    
    def __init__(self, base_dir, checkpoint_dir, h5_path, output_dir, 
                 use_wandb=True, wandb_project='bracs-batch-analysis', wandb_run_name=None):
        self.base_dir = base_dir
        self.checkpoint_dir = checkpoint_dir
        self.h5_path = h5_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create master output directory
        self.master_output = os.path.join(output_dir, f"batch_analysis_{self.timestamp}")
        os.makedirs(self.master_output, exist_ok=True)
        
        # Setup subdirectories
        self.dirs = {
            'individual_slides': os.path.join(self.master_output, "individual_slides"),
            'aggregated_metrics': os.path.join(self.master_output, "aggregated_metrics"),
            'reports': os.path.join(self.master_output, "reports"),
            'visualizations': os.path.join(self.master_output, "visualizations"),
            'logs': os.path.join(self.master_output, "logs")
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize results storage
        self.all_results = {}
        self.failed_slides = []
        self.annotation_stats = []
        self.class_distribution = {}
        
        # Setup logging
        self.setup_logging()
        
        # Initialize WandB
        self.use_wandb = use_wandb
        if self.use_wandb:
            run_name = wandb_run_name or f"bracs_batch_{self.timestamp}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    'base_dir': base_dir,
                    'checkpoint_dir': checkpoint_dir,
                    'timestamp': self.timestamp,
                    'output_dir': self.master_output
                }
            )
        
        self.logger.info("="*80)
        self.logger.info("BRACS Batch Analysis Processor")
        self.logger.info("="*80)
        self.logger.info(f"Output directory: {self.master_output}")
        self.logger.info(f"WandB enabled: {self.use_wandb}")
    
    def setup_logging(self):
        """Setup logging for batch processing"""
        self.logger = logging.getLogger('BRACSBatch')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        fh = logging.FileHandler(
            os.path.join(self.dirs['logs'], f'batch_analysis_{self.timestamp}.log')
        )
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def get_slide_list(self):
        """Get list of slides with both SVS and GeoJSON files"""
        svs_files = set()
        geojson_files = set()
        
        # Check HDF5 for available features
        h5_slides = set()
        try:
            with h5py.File(self.h5_path, 'r') as h5_file:
                h5_slides = set(h5_file.keys())
        except Exception as e:
            self.logger.error(f"Could not read HDF5 file: {e}")
            return []
        
        # Find slides with SVS files
        for file in os.listdir(self.base_dir):
            if file.endswith('.svs'):
                slide_id = file.replace('.svs', '')
                if slide_id in h5_slides:
                    svs_files.add(slide_id)
            elif file.endswith('.geojson'):
                slide_id = file.replace('.geojson', '')
                if slide_id in h5_slides:
                    geojson_files.add(slide_id)
        
        # Get slides with both SVS and features (annotations optional)
        valid_slides_with_annotations = sorted(list(svs_files.intersection(geojson_files)))
        valid_slides_without_annotations = sorted(list(svs_files - geojson_files))
        
        self.logger.info(f"Found {len(valid_slides_with_annotations)} slides with annotations")
        self.logger.info(f"Found {len(valid_slides_without_annotations)} slides without annotations")
        
        # Show examples
        if valid_slides_with_annotations:
            self.logger.info("\nExamples with annotations:")
            for slide in valid_slides_with_annotations[:3]:
                geojson_size = os.path.getsize(os.path.join(self.base_dir, f"{slide}.geojson"))
                self.logger.info(f"  - {slide} (annotation size: {geojson_size/1024:.1f}KB)")
        
        return valid_slides_with_annotations, valid_slides_without_annotations
    
    def analyze_annotation_distribution(self, slides):
        """Analyze the distribution of annotations across slides"""
        self.logger.info("\nAnalyzing annotation distribution...")
        
        annotation_stats = []
        class_totals = {'benign': 0, 'atypical': 0, 'malignant': 0, 'other': 0}
        
        for slide_id in slides:
            geojson_path = os.path.join(self.base_dir, f"{slide_id}.geojson")
            
            try:
                with open(geojson_path, 'r') as f:
                    geo_data = json.load(f)
                
                class_counts = {'benign': 0, 'atypical': 0, 'malignant': 0, 'other': 0}
                
                for feature in geo_data.get('features', []):
                    if feature['geometry']['type'] == 'Polygon':
                        class_name = feature.get('properties', {}).get('classification', {}).get('name', 'unknown').lower()
                        
                        if 'malignant' in class_name or 'cancer' in class_name or 'carcinoma' in class_name:
                            class_counts['malignant'] += 1
                            class_totals['malignant'] += 1
                        elif 'dcis' in class_name or 'udh' in class_name or 'atypical' in class_name:
                            class_counts['atypical'] += 1
                            class_totals['atypical'] += 1
                        elif 'benign' in class_name or 'normal' in class_name:
                            class_counts['benign'] += 1
                            class_totals['benign'] += 1
                        else:
                            class_counts['other'] += 1
                            class_totals['other'] += 1
                
                total = sum(class_counts.values())
                annotation_stats.append({
                    'slide_id': slide_id,
                    'total_annotations': total,
                    **class_counts
                })
                
            except Exception as e:
                self.logger.error(f"Error reading annotations for {slide_id}: {e}")
        
        self.class_distribution = class_totals
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({
                'total_benign_annotations': class_totals['benign'],
                'total_atypical_annotations': class_totals['atypical'],
                'total_malignant_annotations': class_totals['malignant'],
                'total_other_annotations': class_totals['other']
            })
        
        return pd.DataFrame(annotation_stats)
    
    def process_single_slide(self, slide_id, model_paths, dtfd_paths, has_annotation=True):
        """Process a single slide with error handling"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {slide_id} (annotation: {has_annotation})")
        self.logger.info(f"{'='*60}")
        
        slide_path = os.path.join(self.base_dir, f"{slide_id}.svs")
        annotation_path = os.path.join(self.base_dir, f"{slide_id}.geojson") if has_annotation else None
        
        try:
            # Initialize generator for this slide (without individual WandB)
            generator = BRACSHeatmapGenerator(
                h5_path=self.h5_path,
                slide_path=slide_path,
                checkpoint_dir=self.checkpoint_dir,
                annotation_path=annotation_path,
                output_dir=os.path.join(self.dirs['individual_slides'], slide_id),
                use_wandb=False  # Batch analyzer handles WandB
            )
            
            # Reduce verbosity for batch processing
            generator.logger.setLevel(logging.WARNING)
            
            # Process slide
            heatmaps = generator.process_slide(slide_id, model_paths, dtfd_paths)
            
            if heatmaps:
                # Extract metrics if available
                metrics_file = os.path.join(generator.dirs['reports'], f'{slide_id}_quantitative_metrics.csv')
                
                if os.path.exists(metrics_file):
                    metrics_df = pd.read_csv(metrics_file, index_col=0)
                    self.all_results[slide_id] = {
                        'metrics': metrics_df,
                        'models_processed': list(heatmaps.keys()),
                        'has_annotation': has_annotation
                    }
                    
                    # Log to WandB
                    if self.use_wandb:
                        for model_name in metrics_df.index:
                            for metric_name in metrics_df.columns:
                                value = metrics_df.loc[model_name, metric_name]
                                if not pd.isna(value):
                                    wandb.log({
                                        f"batch_{slide_id}_{model_name}_{metric_name}": value
                                    })
                    
                    self.logger.info(f"✓ Successfully processed {slide_id} with metrics")
                else:
                    # Store basic info even if no quantitative metrics
                    self.all_results[slide_id] = {
                        'models_processed': list(heatmaps.keys()),
                        'has_annotation': has_annotation
                    }
                    self.logger.info(f"✓ Processed {slide_id} (no quantitative metrics)")
                
                return True
            
            self.logger.warning(f"✗ Failed to generate heatmaps for {slide_id}")
            self.failed_slides.append(slide_id)
            return False
            
        except Exception as e:
            self.logger.error(f"✗ Error processing {slide_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.failed_slides.append(slide_id)
            return False
    
    def aggregate_results(self):
        """Aggregate results across all slides"""
        if not self.all_results:
            self.logger.warning("No results to aggregate!")
            return None, None
        
        # Combine all metrics
        all_metrics = []
        for slide_id, result_data in self.all_results.items():
            if 'metrics' in result_data:
                metrics_df = result_data['metrics']
                for model_name in metrics_df.index:
                    row_data = {
                        'slide_id': slide_id, 
                        'model': model_name,
                        'has_annotation': result_data['has_annotation']
                    }
                    row_data.update(metrics_df.loc[model_name].to_dict())
                    all_metrics.append(row_data)
        
        if not all_metrics:
            self.logger.warning("No quantitative metrics to aggregate")
            return None, None
        
        combined_df = pd.DataFrame(all_metrics)
        
        # Calculate statistics per model
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['slide_id']]
        
        if numeric_cols:
            model_stats = combined_df.groupby('model')[numeric_cols].agg(['mean', 'std', 'median']).round(4)
            # Flatten column names
            model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
        else:
            model_stats = None
        
        return combined_df, model_stats
    
    def create_visualization_summary(self, combined_df):
        """Create summary visualizations"""
        if combined_df is None or combined_df.empty:
            self.logger.warning("No data for visualization")
            return None
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')
        
        sns.set_palette("husl")
        
        # Check available metrics
        available_metrics = []
        for metric in ['auc_malignant', 'auc_atypical', 'auc_abnormal', 
                      'dice_malignant', 'dice_atypical', 'dice_abnormal',
                      'iou_malignant', 'iou_atypical', 'iou_abnormal',
                      'entropy', 'attention_coverage']:
            if metric in combined_df.columns:
                available_metrics.append(metric)
        
        if not available_metrics:
            self.logger.warning("No metrics available for visualization")
            return None
        
        # Create comprehensive figure
        n_metrics = min(len(available_metrics), 6)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(available_metrics[:n_metrics]):
            ax = axes[idx]
            
            # Create boxplot for each model
            models = combined_df['model'].unique()
            data_to_plot = []
            labels = []
            
            for model in models:
                model_data = combined_df[combined_df['model'] == model][metric].dropna()
                if len(model_data) > 0:
                    data_to_plot.append(model_data.values)
                    labels.append(model)
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Color boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                ax.set_xlabel('Model')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                
                # Rotate labels if many models
                if len(labels) > 4:
                    ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('BRACS Multi-Slide Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        viz_path = os.path.join(self.dirs['visualizations'], 'performance_summary.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Visualization saved to: {viz_path}")
        
        if self.use_wandb:
            wandb.log({"performance_summary": wandb.Image(viz_path)})
        
        return viz_path
    
    def create_class_distribution_plot(self):
        """Create visualization of annotation class distribution"""
        if not self.class_distribution:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        labels = list(self.class_distribution.keys())
        sizes = list(self.class_distribution.values())
        colors = ['green', 'yellow', 'red', 'gray']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Annotation Class Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        ax2.bar(labels, sizes, color=colors)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_title('Annotation Counts by Class', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('BRACS Dataset Composition', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        viz_path = os.path.join(self.dirs['visualizations'], 'class_distribution.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.use_wandb:
            wandb.log({"class_distribution": wandb.Image(viz_path)})
        
        return viz_path
    
    def generate_master_report(self, annotation_stats, model_stats):
        """Generate comprehensive report for all slides"""
        report_path = os.path.join(self.dirs['reports'], 'MASTER_ANALYSIS_REPORT.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BATCH ANALYSIS REPORT FOR BRACS DATASET\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Slides Processed: {len(self.all_results)}\n")
            f.write(f"Failed Slides: {len(self.failed_slides)}\n")
            f.write(f"Slides with Annotations: {sum(1 for r in self.all_results.values() if r['has_annotation'])}\n\n")
            
            # Dataset composition
            if self.class_distribution:
                f.write("DATASET COMPOSITION:\n")
                f.write("-"*40 + "\n")
                for class_name, count in self.class_distribution.items():
                    f.write(f"  {class_name.capitalize()}: {count} annotations\n")
                f.write("\n")
            
            if annotation_stats is not None and not annotation_stats.empty:
                f.write("ANNOTATION STATISTICS:\n")
                f.write("-"*40 + "\n")
                f.write(f"  Mean annotations per slide: {annotation_stats['total_annotations'].mean():.1f}\n")
                f.write(f"  Max annotations in a slide: {annotation_stats['total_annotations'].max()}\n")
                f.write(f"  Min annotations in a slide: {annotation_stats['total_annotations'].min()}\n\n")
            
            # Model performance summary
            if model_stats is not None:
                f.write("MODEL PERFORMANCE SUMMARY:\n")
                f.write("-"*40 + "\n\n")
                
                # Find best performing models
                ranking_metrics = []
                if 'auc_malignant_mean' in model_stats.columns:
                    ranking_metrics.append('auc_malignant_mean')
                if 'auc_abnormal_mean' in model_stats.columns:
                    ranking_metrics.append('auc_abnormal_mean')
                if 'dice_malignant_mean' in model_stats.columns:
                    ranking_metrics.append('dice_malignant_mean')
                
                for metric in ranking_metrics:
                    f.write(f"\nRanking by {metric}:\n")
                    rankings = model_stats.sort_values(metric, ascending=False)
                    for i, (model, row) in enumerate(rankings.iterrows(), 1):
                        mean_val = row[metric]
                        std_val = row[metric.replace('_mean', '_std')] if metric.replace('_mean', '_std') in row else 0
                        f.write(f"  {i}. {model}: {mean_val:.4f} ± {std_val:.4f}\n")
                
                # Detailed statistics
                f.write("\n\nDETAILED MODEL STATISTICS:\n")
                f.write("-"*40 + "\n")
                f.write(model_stats.to_string())
                f.write("\n")
            
            # Failed slides
            if self.failed_slides:
                f.write("\n\nFAILED SLIDES:\n")
                f.write("-"*40 + "\n")
                for slide in self.failed_slides:
                    f.write(f"  - {slide}\n")
            
            # Processing statistics
            f.write("\n\nPROCESSING STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"  Success rate: {len(self.all_results)/(len(self.all_results)+len(self.failed_slides))*100:.1f}%\n")
            f.write(f"  Output directory: {self.master_output}\n")
        
        self.logger.info(f"\n✓ Master report saved to: {report_path}")
        
        if self.use_wandb:
            wandb.save(report_path)
        
        return report_path
    
    def save_aggregated_data(self, combined_df, model_stats, annotation_stats):
        """Save all aggregated data to CSV files"""
        
        # Save combined metrics
        if combined_df is not None:
            combined_path = os.path.join(self.dirs['aggregated_metrics'], 'all_metrics_combined.csv')
            combined_df.to_csv(combined_path, index=False)
            self.logger.info(f"Saved combined metrics to: {combined_path}")
            
            if self.use_wandb:
                wandb.save(combined_path)
        
        # Save model statistics
        if model_stats is not None:
            stats_path = os.path.join(self.dirs['aggregated_metrics'], 'model_statistics.csv')
            model_stats.to_csv(stats_path)
            self.logger.info(f"Saved model statistics to: {stats_path}")
            
            if self.use_wandb:
                wandb.save(stats_path)
        
        # Save annotation statistics
        if annotation_stats is not None and not annotation_stats.empty:
            ann_path = os.path.join(self.dirs['aggregated_metrics'], 'annotation_statistics.csv')
            annotation_stats.to_csv(ann_path, index=False)
            self.logger.info(f"Saved annotation statistics to: {ann_path}")
            
            if self.use_wandb:
                wandb.save(ann_path)

def main():
    """Main execution function"""
    
    # Configuration
    config = {
        'base_dir': '/vol/research/scratch1/NOBACKUP/rk01337/BRACS/wsi_slide',
        'checkpoint_dir': '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/checkpoint_analysis/BRACS/resnet18',
        'h5_path': '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/processed_bracs_10_filtered/patch_feats_pretrain_natural_supervised_Resnet18.h5',
        'output_dir': '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/bracs_batch_results'
    }
    
    # Model paths
    model_paths = {
        'acmil': os.path.join(config['checkpoint_dir'], 'bracs_acmil_resnet18_checkpoint-best.pth'),
        'abmil': os.path.join(config['checkpoint_dir'], 'bracs_abmil_resnet18_checkpoint-best.pth'),
        'transmil': os.path.join(config['checkpoint_dir'], 'bracs_transmil_resnet18_checkpoint-best.pth'),
        'meanmil': os.path.join(config['checkpoint_dir'], 'bracs_meanmil_resnet18_checkpoint-best.pth'),
        'maxmil': os.path.join(config['checkpoint_dir'], 'bracs_maxmil_resnet18_checkpoint-best.pth'),
        'oodml': os.path.join(config['checkpoint_dir'], 'bracs_oodml_resnet18_checkpoint-best.pt'),
    }
    
    dtfd_paths = {
        'classifier': os.path.join(config['checkpoint_dir'], 'bracs_dtfd_resnet18_checkpoint-best_classifier.pth'),
        'attention': os.path.join(config['checkpoint_dir'], 'bracs_dtfd_resnet18_checkpoint-best_attention.pth'),
        'dimReduction': os.path.join(config['checkpoint_dir'], 'bracs_dtfd_resnet18_checkpoint-best_dimReduction.pth'),
        'attCls': os.path.join(config['checkpoint_dir'], 'bracs_dtfd_resnet18_checkpoint-best_attCls.pth'),
    }
    
    # Initialize batch analyzer with WandB
    analyzer = BatchBRACSAnalyzer(
        config['base_dir'],
        config['checkpoint_dir'],
        config['h5_path'],
        config['output_dir'],
        use_wandb=True,
        wandb_project='bracs-batch-analysis',
        wandb_run_name=f'bracs_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Get slide lists
    slides_with_ann, slides_without_ann = analyzer.get_slide_list()
    
    if not slides_with_ann and not slides_without_ann:
        print("No valid slides found!")
        return
    
    # Analyze annotation distribution for slides with annotations
    annotation_stats = None
    if slides_with_ann:
        print("\nAnalyzing annotation distribution...")
        annotation_stats = analyzer.analyze_annotation_distribution(slides_with_ann)
        
        if not annotation_stats.empty:
            print("\nAnnotation Summary:")
            print(annotation_stats.describe())
    
    # Combine all slides
    all_slides = slides_with_ann + slides_without_ann
    
    # User confirmation
    print(f"\n{'='*60}")
    print(f"BRACS BATCH ANALYSIS")
    print(f"{'='*60}")
    print(f"Total slides to process: {len(all_slides)}")
    print(f"  - With annotations: {len(slides_with_ann)}")
    print(f"  - Without annotations: {len(slides_without_ann)}")
    print(f"Output directory: {analyzer.master_output}")
    print("\nAnalysis will include:")
    print("  - Individual heatmaps for each model")
    print("  - Consolidated heatmap figures (pure heatmaps)")
    print("  - Consolidated comparison figures (with overlays)")
    print("  - Quantitative metrics (Dice, IoU, AUC) where annotations available")
    print("  - Aggregated statistics across all slides")
    print("\nRecommendation: Run this in a screen/tmux session to avoid interruption")
    print("="*60)
    
    response = input(f"\nProceed with processing {len(all_slides)} slides? (yes/no): ")
    if response.lower() != 'yes':
        print("Batch processing cancelled.")
        if analyzer.use_wandb:
            wandb.finish()
        return
    
    # Process all slides
    print(f"\nStarting batch processing of {len(all_slides)} slides...")
    successful = 0
    start_time = datetime.now()
    
    # Process slides with annotations first
    for i, slide_id in enumerate(slides_with_ann, 1):
        print(f"\n[{i}/{len(all_slides)}] Processing {slide_id} (with annotations)...")
        
        try:
            if analyzer.process_single_slide(slide_id, model_paths, dtfd_paths, has_annotation=True):
                successful += 1
            
            # Progress update
            if i % 5 == 0 or i == len(slides_with_ann):
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(all_slides) - i) / rate if rate > 0 else 0
                print(f"\nProgress: {i}/{len(all_slides)} ({100*i/len(all_slides):.1f}%)")
                print(f"Successful: {successful}/{i} ({100*successful/i:.1f}%)")
                print(f"Elapsed: {elapsed:.1f}m, Estimated remaining: {remaining:.1f}m")
                
                # Log to WandB
                if analyzer.use_wandb:
                    wandb.log({
                        'progress': i / len(all_slides),
                        'success_rate': successful / i,
                        'elapsed_minutes': elapsed
                    })
        
        except KeyboardInterrupt:
            print(f"\nProcessing interrupted at slide {i}")
            print(f"Partial results will be saved...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
    
    # Process slides without annotations
    offset = len(slides_with_ann)
    for i, slide_id in enumerate(slides_without_ann, offset + 1):
        print(f"\n[{i}/{len(all_slides)}] Processing {slide_id} (no annotations)...")
        
        try:
            if analyzer.process_single_slide(slide_id, model_paths, dtfd_paths, has_annotation=False):
                successful += 1
            
            # Progress update
            if i % 5 == 0 or i == len(all_slides):
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(all_slides) - i) / rate if rate > 0 else 0
                print(f"\nProgress: {i}/{len(all_slides)} ({100*i/len(all_slides):.1f}%)")
                print(f"Successful: {successful}/{i} ({100*successful/i:.1f}%)")
                print(f"Elapsed: {elapsed:.1f}m, Estimated remaining: {remaining:.1f}m")
                
                # Log to WandB
                if analyzer.use_wandb:
                    wandb.log({
                        'progress': i / len(all_slides),
                        'success_rate': successful / i,
                        'elapsed_minutes': elapsed
                    })
        
        except KeyboardInterrupt:
            print(f"\nProcessing interrupted at slide {i}")
            print(f"Partial results will be saved...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
    
    # Calculate final statistics
    end_time = datetime.now()
    total_elapsed = (end_time - start_time).total_seconds() / 60
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Successfully processed: {successful}/{len(all_slides)} slides")
    print(f"Total time: {total_elapsed:.1f} minutes ({total_elapsed/60:.1f} hours)")
    if successful > 0:
        print(f"Average time per slide: {total_elapsed/successful:.2f} minutes")
    print(f"{'='*60}")
    
    # Aggregate results
    if analyzer.all_results:
        print("\nAggregating results...")
        combined_df, model_stats = analyzer.aggregate_results()
        
        # Save aggregated data
        analyzer.save_aggregated_data(combined_df, model_stats, annotation_stats)
        
        # Generate master report
        analyzer.generate_master_report(annotation_stats, model_stats)
        
        # Create visualizations
        if combined_df is not None:
            analyzer.create_visualization_summary(combined_df)
        
        # Create class distribution plot
        if analyzer.class_distribution:
            analyzer.create_class_distribution_plot()
        
        print(f"\n✓ All results saved to: {analyzer.master_output}")
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Slides processed: {successful}")
        print(f"  - With annotations: {sum(1 for r in analyzer.all_results.values() if r['has_annotation'])}")
        print(f"  - Without annotations: {sum(1 for r in analyzer.all_results.values() if not r['has_annotation'])}")
        
        if model_stats is not None:
            print(f"Models evaluated: {len(model_stats)}")
            
            # Print top model if AUC metrics available
            if 'auc_malignant_mean' in model_stats.columns:
                best_model = model_stats['auc_malignant_mean'].idxmax()
                print(f"\nTop performing model (by mean AUC-Malignant):")
                print(f"  {best_model}: {model_stats.loc[best_model, 'auc_malignant_mean']:.4f}")
            elif 'auc_abnormal_mean' in model_stats.columns:
                best_model = model_stats['auc_abnormal_mean'].idxmax()
                print(f"\nTop performing model (by mean AUC-Abnormal):")
                print(f"  {best_model}: {model_stats.loc[best_model, 'auc_abnormal_mean']:.4f}")
        
        print(f"\nOutput directory: {analyzer.master_output}")
        
        # Final WandB logging
        if analyzer.use_wandb:
            wandb.log({
                'total_slides_processed': successful,
                'total_slides_failed': len(analyzer.failed_slides),
                'total_time_minutes': total_elapsed,
                'average_time_per_slide': total_elapsed/successful if successful > 0 else 0
            })
    else:
        print("\nNo results to aggregate. Check error logs.")
    
    if analyzer.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
