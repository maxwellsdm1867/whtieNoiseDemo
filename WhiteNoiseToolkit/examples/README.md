# White Noise Analysis Toolkit - Examples

This directory contains example scripts demonstrating the functionality of the White Noise Analysis Toolkit.

## Available Examples

### 1. Simple Demo (`simple_demo.py`)
**Purpose**: Demonstrates basic synthetic data generation and visualization.

**What it does**:
- Creates a realistic biphasic neural filter (ground truth)
- Defines a sigmoid nonlinearity function
- Generates synthetic white noise stimulus
- Simulates neural responses using the toolkit
- Creates comprehensive visualizations of all components

**Usage**:
```bash
cd examples
python simple_demo.py
```

**Output**:
- Interactive plots showing ground truth parameters and synthetic data
- Saved figures: `demo_output/synthetic_data_demo.png` and `.pdf`

### 2. Complete Analysis Demo (`complete_analysis_demo.py`)
**Purpose**: Full analysis workflow including filter and nonlinearity recovery.

**What it does**:
- Generates synthetic data with known ground truth
- Adds realistic noise to neural responses
- Uses the toolkit to recover filters and nonlinearities
- Compares recovered parameters with ground truth
- Provides comprehensive quality assessment

**Usage**:
```bash
cd examples
python complete_analysis_demo.py
```

**Output**:
- Multi-panel visualization comparing true vs. recovered parameters
- Performance metrics and quality assessment
- Saved figures: `demo_output/complete_analysis_demo.png` and `.pdf`

## Key Features Demonstrated

### Synthetic Data Generation
- ✅ Realistic biphasic temporal filters
- ✅ Sigmoid nonlinearity with threshold and saturation
- ✅ White noise stimulus generation
- ✅ Neural response simulation with noise

### Visualization
- ✅ Ground truth filter visualization
- ✅ Nonlinearity curves
- ✅ Stimulus and spike raster plots
- ✅ Power spectrum analysis
- ✅ Statistical summaries

### Quality Assessment
- ✅ Filter correlation analysis
- ✅ Signal-to-noise ratio calculations
- ✅ Firing rate statistics
- ✅ Recovery performance metrics

## Example Output

The demos generate:
1. **Filter visualization**: Shows biphasic temporal profiles
2. **Nonlinearity plots**: Sigmoid response functions
3. **Stimulus samples**: White noise time series
4. **Spike rasters**: Neural response patterns
5. **Summary statistics**: Data quality metrics

## Parameters

### Typical Settings
- **Duration**: 2-5 minutes of data
- **Sampling rate**: 125 Hz (8ms bins)
- **Filter length**: 50 samples (400ms)
- **Noise level**: 0.3 (moderate noise)

### Customization
You can modify parameters in the scripts:
```python
duration_minutes = 3.0
sampling_rate = 1000.0
noise_level = 0.3
filter_length = 50
```

## Next Steps

After running these examples, you can:

1. **Experiment with parameters**: Try different noise levels, filter shapes, or data durations
2. **Test real data**: Apply the toolkit to your own electrophysiology recordings
3. **Extend analysis**: Add spatial components or multi-electrode analysis
4. **Validate methods**: Use synthetic data to benchmark algorithm performance

## Requirements

The examples require:
- NumPy, SciPy, Matplotlib, Seaborn
- White Noise Analysis Toolkit (installed in parent directory)
- Python 3.8+

## Output Directory

Results are saved to `demo_output/` which is automatically created when running the examples.

## Support

For questions about the examples or toolkit functionality, refer to:
- Main README.md in the project root
- API documentation in the source code
- Tutorials in `docs/tutorials/`
