#!/usr/bin/env python3
"""
Batch analysis script for CAMELYON16 dataset with multi-threshold analysis.
Finds best threshold for each model-slide combination.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import wandb

# Add the path to your existing CAMELYON script
sys.path.append('/vol/research/scratch1/NOBACKUP/rk01337/ACMIL')

# Import the generator class
from generate_heatmap_all_models_camelyon import CAMELYONHeatmapGenerator

class CAMELYONBatchAnalyzer:
    """
    Batch analyzer for CAMELYON16 dataset with multi-threshold analysis
    """
    
    def __init__(self, config, use_wandb=True):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create master output directory
        self.master_output = os.path.join(
            config['output_dir'], 
            f"batch_analysis_{self.timestamp}"
        )
        os.makedirs(self.master_output, exist_ok=True)
        
        # Setup subdirectories
        self.dirs = {
            'individual_slides': os.path.join(self.master_output, "individual_slides"),
            'aggregated_metrics': os.path.join(self.master_output, "aggregated_metrics"),
            'reports': os.path.join(self.master_output, "reports"),
            'visualizations': os.path.join(self.master_output, "visualizations"),
            'threshold_analysis': os.path.join(self.master_output, "threshold_analysis")
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize results storage
        self.all_results = {}
        self.failed_slides = []
        self.tumor_burden = {}
        self.threshold_info = {}  # Track thresholds used
        
        # Setup logging
        self.setup_logging()
        
        # Initialize WandB
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(
                project='camelyon-batch-multithresh',
                name=f'batch_multithresh_{self.timestamp}',
                config={
                    'thresholding': 'multi-threshold',
                    'timestamp': self.timestamp,
                    'output_dir': self.master_output
                }
            )
    
    def setup_logging(self):
        """Setup logging for batch processing"""
        self.logger = logging.getLogger('CAMELYONBatch')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        fh = logging.FileHandler(
            os.path.join(self.master_output, f'batch_analysis_{self.timestamp}.log')
        )
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def get_available_slides(self):
        """Get list of slides with all required data"""
        slide_dir = self.config['slide_dir']
        annotation_dir = self.config['annotation_dir']
        h5_path = self.config['h5_path']
        
        # Get available slides from HDF5
        with h5py.File(h5_path, 'r') as h5_file:
            h5_slides = set(h5_file.keys())
        
        # Get slides with annotations (all tumor slides)
        annotated_slides = set()
        for file in os.listdir(annotation_dir):
            if file.endswith('.xml'):
                slide_id = file.replace('.xml', '')
                annotated_slides.add(slide_id)
        
        # Get slides with images
        image_slides = set()
        for file in os.listdir(slide_dir):
            if file.endswith('.tif'):
                slide_id = file.replace('.tif', '')
                image_slides.add(slide_id)
        
        # Find intersection
        valid_slides = h5_slides.intersection(annotated_slides).intersection(image_slides)
        
        self.logger.info(f"Found {len(valid_slides)} valid tumor slides with all required data")
        self.logger.info(f"  HDF5 slides: {len(h5_slides)}")
        self.logger.info(f"  Annotated slides: {len(annotated_slides)}")
        self.logger.info(f"  Image slides: {len(image_slides)}")
        
        return sorted(list(valid_slides))
    
    def process_single_slide(self, slide_id, generator, model_paths, dtfd_paths):
        """Process a single slide using the generator"""
        try:
            self.logger.info(f"Processing tumor slide: {slide_id}")
            
            # Use the generator to process the slide
            metrics = generator.process_slide(slide_id, model_paths, dtfd_paths)
            
            if metrics:
                # Store results
                self.all_results[slide_id] = metrics
                
                # Extract tumor burden and threshold info
                for model_name, model_metrics in metrics.items():
                    if 'tumor_burden' in model_metrics:
                        self.tumor_burden[slide_id] = model_metrics['tumor_burden']
                    
                    # Track threshold used
                    if slide_id not in self.threshold_info:
                        self.threshold_info[slide_id] = {}
                    self.threshold_info[slide_id][model_name] = {
                        'type': model_metrics.get('threshold_type', 'unknown'),
                        'value': model_metrics.get('threshold_value', 0)
                    }
                
                # Save individual slide metrics
                slide_metrics_path = os.path.join(
                    self.dirs['individual_slides'], 
                    f'{slide_id}_metrics.json'
                )
                with open(slide_metrics_path, 'w') as f:
                    json.dump({
                        'slide_id': slide_id,
                        'tumor_burden': self.tumor_burden.get(slide_id, None),
                        'metrics': metrics,
                        'thresholds_used': self.threshold_info.get(slide_id, {})
                    }, f, indent=2)
                
                self.logger.info(f"✓ Successfully processed {slide_id}")
                if slide_id in self.tumor_burden:
                    self.logger.info(f"  Tumor burden: {self.tumor_burden[slide_id]*100:.1f}%")
                
                # Log to WandB
                if self.use_wandb:
                    for model_name, model_metrics in metrics.items():
                        wandb.log({
                            f"batch_{slide_id}_{model_name}_iou": model_metrics.get('iou', 0),
                            f"batch_{slide_id}_{model_name}_f1": model_metrics.get('f1_score', 0),
                            f"batch_{slide_id}_{model_name}_auc": model_metrics.get('auc_roc', 0)
                        })
                
                return True
            else:
                self.logger.warning(f"No metrics generated for {slide_id}")
                self.failed_slides.append(slide_id)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to process {slide_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.failed_slides.append(slide_id)
            return False
    
    def aggregate_metrics(self):
        """Aggregate metrics across all tumor slides"""
        if not self.all_results:
            self.logger.warning("No results to aggregate")
            return None, None
        
        # Convert to DataFrame
        rows = []
        for slide_id, slide_metrics in self.all_results.items():
            for model_name, metrics in slide_metrics.items():
                row = {
                    'slide_id': slide_id,
                    'model': model_name,
                    'tumor_burden': self.tumor_burden.get(slide_id, None),
                    'threshold_type': metrics.get('threshold_type', 'unknown'),
                    'threshold_value': metrics.get('threshold_value', 0),
                    **metrics
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            return df, pd.DataFrame()
        
        # Calculate summary statistics
        numeric_cols = ['iou', 'dice', 'f1_score', 'sensitivity', 'specificity', 
                       'precision', 'accuracy', 'balanced_accuracy', 'auc_roc', 'avg_precision']
        existing_cols = [col for col in numeric_cols if col in df.columns]
        
        summary_stats = df.groupby('model')[existing_cols].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(4)
        
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        
        return df, summary_stats
    
    def analyze_thresholds(self, df):
        """Analyze threshold patterns across models and slides"""
        threshold_analysis = {}
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            # Count threshold types used
            threshold_counts = model_df['threshold_type'].value_counts()
            
            # Get threshold value statistics
            threshold_stats = model_df['threshold_value'].describe()
            
            threshold_analysis[model] = {
                'threshold_types': threshold_counts.to_dict(),
                'threshold_value_mean': threshold_stats['mean'],
                'threshold_value_std': threshold_stats['std'],
                'threshold_value_min': threshold_stats['min'],
                'threshold_value_max': threshold_stats['max']
            }
        
        # Save threshold analysis
        threshold_df = pd.DataFrame(threshold_analysis).T
        threshold_df.to_csv(os.path.join(self.dirs['threshold_analysis'], 'threshold_patterns.csv'))
        
        return threshold_analysis
    
    def create_visualizations(self, df, summary_stats):
        """Create comprehensive visualizations"""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Performance comparison with best thresholds
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics_to_plot = ['iou', 'f1_score', 'sensitivity', 'precision', 'auc_roc', 'avg_precision']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            if f'{metric}_mean' in summary_stats.columns:
                models = summary_stats.index
                means = summary_stats[f'{metric}_mean']
                stds = summary_stats[f'{metric}_std']
                
                bars = ax.bar(range(len(models)), means, yerr=stds, capsize=5)
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45)
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} (Mean ± Std)')
                ax.grid(True, alpha=0.3)
                
                # Color best performer
                best_idx = np.argmax(means)
                bars[best_idx].set_color('green')
        
        plt.suptitle('Model Performance Comparison\n(Multi-threshold Analysis)', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.dirs['visualizations'], 'performance_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.use_wandb:
            wandb.log({"performance_comparison": wandb.Image(save_path)})
        
        # 2. Threshold distribution visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot threshold values by model
        ax = axes[0, 0]
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            ax.hist(model_df['threshold_value'], alpha=0.5, label=model, bins=20)
        ax.set_xlabel('Threshold Value')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Selected Thresholds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot threshold types by model
        ax = axes[0, 1]
        threshold_type_counts = df.groupby(['model', 'threshold_type']).size().unstack(fill_value=0)
        threshold_type_counts.T.plot(kind='bar', ax=ax)
        ax.set_xlabel('Threshold Type')
        ax.set_ylabel('Count')
        ax.set_title('Threshold Types Used by Model')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        
        # Plot IoU vs threshold value
        ax = axes[1, 0]
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            ax.scatter(model_df['threshold_value'], model_df['iou'], alpha=0.6, label=model, s=50)
        ax.set_xlabel('Threshold Value')
        ax.set_ylabel('IoU')
        ax.set_title('IoU vs Threshold Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot performance by tumor burden
        ax = axes[1, 1]
        if 'tumor_burden' in df.columns and df['tumor_burden'].notna().any():
            df['burden_category'] = pd.cut(df['tumor_burden'], 
                                          bins=[0, 0.01, 0.05, 0.1, 1.0],
                                          labels=['<1%', '1-5%', '5-10%', '>10%'])
            burden_performance = df.groupby(['model', 'burden_category'])['iou'].mean().unstack()
            burden_performance.T.plot(kind='bar', ax=ax)
            ax.set_xlabel('Tumor Burden Category')
            ax.set_ylabel('Mean IoU')
            ax.set_title('Performance by Tumor Burden')
            ax.legend(title='Model')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Threshold Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.dirs['visualizations'], 'threshold_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.use_wandb:
            wandb.log({"threshold_analysis": wandb.Image(save_path)})
        
        self.logger.info(f"Created visualizations in {self.dirs['visualizations']}")
    
    def generate_report(self, df, summary_stats, threshold_analysis):
        """Generate comprehensive analysis report"""
        report_path = os.path.join(self.dirs['reports'], 'analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CAMELYON16 TUMOR DETECTION ANALYSIS REPORT\n")
            f.write("Multi-threshold Analysis\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Slides Processed: {len(self.all_results)}\n")
            f.write(f"Failed Slides: {len(self.failed_slides)}\n\n")
            
            # Tumor burden statistics
            if self.tumor_burden:
                burdens = list(self.tumor_burden.values())
                f.write("TUMOR BURDEN STATISTICS:\n")
                f.write("-"*40 + "\n")
                f.write(f"  Mean tumor burden: {np.mean(burdens)*100:.1f}%\n")
                f.write(f"  Median tumor burden: {np.median(burdens)*100:.1f}%\n")
                f.write(f"  Min tumor burden: {np.min(burdens)*100:.1f}%\n")
                f.write(f"  Max tumor burden: {np.max(burdens)*100:.1f}%\n\n")
            
            # Model performance summary
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("-"*40 + "\n\n")
            
            if 'iou_mean' in summary_stats.columns:
                ranked_by_iou = summary_stats.nlargest(len(summary_stats), 'iou_mean')
                f.write("Ranking by IoU:\n")
                for i, model in enumerate(ranked_by_iou.index, 1):
                    f.write(f"  {i}. {model}: IoU={ranked_by_iou.loc[model, 'iou_mean']:.4f} ± "
                           f"{ranked_by_iou.loc[model, 'iou_std']:.4f}\n")
            
            f.write("\n")
            
            # Threshold analysis
            f.write("THRESHOLD ANALYSIS:\n")
            f.write("-"*40 + "\n\n")
            
            for model, info in threshold_analysis.items():
                f.write(f"{model}:\n")
                f.write(f"  Threshold types used: {info['threshold_types']}\n")
                f.write(f"  Mean threshold value: {info['threshold_value_mean']:.3f}\n")
                f.write(f"  Threshold range: [{info['threshold_value_min']:.3f}, {info['threshold_value_max']:.3f}]\n\n")
            
            # Detailed statistics
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED STATISTICS:\n")
            f.write("="*80 + "\n\n")
            
            for model in summary_stats.index:
                f.write(f"\n{model}:\n")
                f.write("-"*len(model) + "\n")
                
                metrics_to_report = ['iou', 'f1_score', 'sensitivity', 'precision', 'auc_roc', 'avg_precision']
                
                for metric in metrics_to_report:
                    if f'{metric}_mean' in summary_stats.columns:
                        mean = summary_stats.loc[model, f'{metric}_mean']
                        std = summary_stats.loc[model, f'{metric}_std']
                        f.write(f"  {metric:15s}: {mean:.4f} ± {std:.4f}\n")
        
        self.logger.info(f"Report saved to: {report_path}")
        
        if self.use_wandb:
            wandb.save(report_path)
        
        return report_path

def main():
    """Main execution function for batch analysis"""
    
    # Configuration
    config = {
        'h5_path': '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/processed_camelyon_20/patch_feats_pretrain_medical_ssl_ViT-S_16.h5',
        'slide_dir': '/vol/research/datasets/pathology/Camelyon/Camelyon16/testing/images',
        'annotation_dir': '/vol/research/datasets/pathology/Camelyon/Camelyon16/testing/annotations',
        'checkpoint_dir': '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/checkpoint_analysis/CAMELYON16/ssl',
        'output_dir': '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/camelyon_batch_multithresh'
    }
    
    # Model paths
    model_paths = {
        'acmil': os.path.join(config['checkpoint_dir'], 'camelyon_acmil_ssl_checkpoint-best.pth'),
        'abmil': os.path.join(config['checkpoint_dir'], 'camelyon_abmil_ssl_checkpoint-best.pth'),
        'transmil': os.path.join(config['checkpoint_dir'], 'camelyon_transmil_ssl_checkpoint-best.pth'),
        'meanmil': os.path.join(config['checkpoint_dir'], 'camelyon_meanmil_ssl_checkpoint-best.pth'),
        'maxmil': os.path.join(config['checkpoint_dir'], 'camelyon_maxmil_ssl_checkpoint-best.pth'),
        'oodml': os.path.join(config['checkpoint_dir'], 'camelyon_oodml_ssl_checkpoint-best.pt'),
    }
    
    # DTFD paths
    dtfd_paths = {
        'classifier': os.path.join(config['checkpoint_dir'], 'camelyon_dtfd_ssl_checkpoint-best_classifier.pth'),
        'attention': os.path.join(config['checkpoint_dir'], 'camelyon_dtfd_ssl_checkpoint-best_attention.pth'),
        'dimReduction': os.path.join(config['checkpoint_dir'], 'camelyon_dtfd_ssl_checkpoint-best_dimReduction.pth'),
        'attCls': os.path.join(config['checkpoint_dir'], 'camelyon_dtfd_ssl_checkpoint-best_attCls.pth'),
    }
    
    # Initialize batch analyzer
    analyzer = CAMELYONBatchAnalyzer(config, use_wandb=True)
    
    # Get available slides
    slides = analyzer.get_available_slides()
    
    analyzer.logger.info(f"\nWill process {len(slides)} tumor slides")
    analyzer.logger.info("Using multi-threshold analysis")
    
    # Initialize generator (without individual WandB run)
    generator = CAMELYONHeatmapGenerator(
        h5_path=config['h5_path'],
        slide_dir=config['slide_dir'],
        checkpoint_dir=config['checkpoint_dir'],
        annotation_dir=config['annotation_dir'],
        output_dir=analyzer.dirs['individual_slides'],
        patch_size=256,
        extraction_level=0,
        use_wandb=False  # Don't create separate WandB run
    )
    
    # Reduce verbosity
    generator.logger.setLevel(logging.WARNING)
    
    # Process slides
    successful = 0
    start_time = datetime.now()
    
    for i, slide_id in enumerate(slides, 1):
        analyzer.logger.info(f"\n[{i}/{len(slides)}] Processing {slide_id}...")
        
        try:
            if analyzer.process_single_slide(slide_id, generator, model_paths, dtfd_paths):
                successful += 1
            
            # Progress update
            if i % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                analyzer.logger.info(f"Progress: {i}/{len(slides)} ({100*i/len(slides):.1f}%)")
                analyzer.logger.info(f"Successful: {successful}/{i}")
                
        except KeyboardInterrupt:
            analyzer.logger.warning(f"\nProcessing interrupted at slide {i}")
            break
    
    # Aggregate and analyze results
    if analyzer.all_results:
        analyzer.logger.info("\nAggregating results...")
        df, summary_stats = analyzer.aggregate_metrics()
        
        if df is not None and not df.empty:
            # Analyze thresholds
            threshold_analysis = analyzer.analyze_thresholds(df)
            
            # Create visualizations
            analyzer.create_visualizations(df, summary_stats)
            
            # Generate report
            analyzer.generate_report(df, summary_stats, threshold_analysis)
            
            # Save aggregated data
            df.to_csv(os.path.join(analyzer.dirs['aggregated_metrics'], 'all_metrics.csv'), index=False)
            summary_stats.to_csv(os.path.join(analyzer.dirs['aggregated_metrics'], 'summary_stats.csv'))
            
            # Log final summary to WandB
            if analyzer.use_wandb:
                for model in summary_stats.index:
                    wandb.log({
                        f"final_{model}_avg_iou": summary_stats.loc[model, 'iou_mean'],
                        f"final_{model}_avg_f1": summary_stats.loc[model, 'f1_score_mean'],
                        f"final_{model}_avg_auc": summary_stats.loc[model, 'auc_roc_mean']
                    })
            
            # Print summary
            analyzer.logger.info("\n" + "="*60)
            analyzer.logger.info("FINAL SUMMARY")
            analyzer.logger.info("="*60)
            analyzer.logger.info(f"Processed: {successful}/{len(slides)} slides")
            
            if 'iou_mean' in summary_stats.columns:
                analyzer.logger.info("\nTop Models by IoU:")
                top_models = summary_stats.nlargest(3, 'iou_mean')
                for i, (model, row) in enumerate(top_models.iterrows(), 1):
                    analyzer.logger.info(f"  {i}. {model}: IoU={row['iou_mean']:.4f}")
    
    analyzer.logger.info(f"\nResults saved in: {analyzer.master_output}")
    
    if analyzer.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
