# ves_vis

This module provides comprehensive visualization tools for vesicle and neuron data. It includes data conversion scripts, mesh generation utilities, and multiple visualization backends including HTML viewers, interactive Plotly plots, PyVista 3D scenes, and Neuroglancer integration.

For setup, install the core dependencies and optional visualization packages as needed:
```bash
pip install -r requirements.txt
```

## Data Conversion

Convert raw vesicle mapping files into structured data formats for visualization.

### Extract Vesicle Data
Parse vesicle mapping files and generate parquet/CSV/JSON output:
```bash
python scripts/conversion/dfGen.py /path/to/mapping/files/ \
    --output vesicles.parquet \
    --types-dir /path/to/label/files/ \
    --compute-neighbors \
    --neighbor-radius 500
```

## Mesh Generation

Generate 3D meshes from volume data and vesicle coordinates.

### Neuron Meshes
Convert HDF5 neuron masks to 3D meshes using marching cubes:
```bash
python scripts/conversion/neuron_mesh_gen.py neuron_mask.h5 neuron.obj \
    --format glb \
    --apply-gaussian-filter \
    --fix-gaps-x-axis
```

### Vesicle Meshes
Generate sphere meshes from vesicle coordinate data:
```bash
python scripts/conversion/vesicle_mesh_gen.py vesicles.parquet \
    --output vesicles.obj \
    --format ply \
    --resolution 6 \
    --color-by type \
    --colormap plasma
```

## Visualization

Multiple visualization backends for different use cases and requirements.

### HTML Viewer
Create self-contained HTML viewer with Three.js:
```bash
python scripts/visualizations/htmlGen.py vesicles.parquet offsets.csv neurons/ colormap.json \
    --output my_viewer \
```

### Interactive Plotly
Generate interactive 3D plots for web browsers:
```bash
python scripts/visualizations/plotlyGen.py vesicles.parquet \
    --neuron-mesh neuron.obj \
    --sample-id SHL17 \
    --color-by type \
    --output interactive_plot.html
```

### PyVista 3D Rendering
Create high-quality 3D visualizations and screenshots:
```bash
python scripts/visualizations/pyvistaGen.py neurons/ vesicles/ offsets.csv \
    --no-interactive \
    --output scene.png \
    --camera-position xy
```

### Neuroglancer Viewer
Launch live Neuroglancer viewer for HDF5 volumes:
```bash
python scripts/visualizations/neuroglancerGen.py neuron_h5_files/ vesicles.h5 offsets.csv \
    --resolution 25 50 50 \
    --keep-open
```

## Complete Workflow Example

Process sample data from raw mapping files to final visualization:

```bash
# 1. Extract vesicle data from mapping files
python scripts/conversion/dfGen.py sample_data/mapping_files/ \
    --output sample_vesicles.parquet \
    --types-dir sample_data/labels/ \
    --compute-neighbors

# 2. Generate neuron mesh from HDF5 mask
python scripts/conversion/neuron_mesh_gen.py sample_data/neuron_SHL17.h5 sample_neuron.glb \
    --format glb

# 3. Create interactive visualization
python scripts/visualizations/plotlyGen.py sample_vesicles.parquet \
    --neuron-mesh sample_neuron.glb \
    --sample-id sample_id \
    --color-by type \
    --output sample_visualization.html
```

Open `sample_visualization.html` in your browser to view the interactive 3D visualization.

## Notes

- **File Formats**: Supports multiple mesh formats (OBJ, PLY, GLB, STL) and data formats (Parquet, CSV, JSON)
- **Coordinate Systems**: Automatically handles X/Z coordinate swapping for proper spatial alignment
- **Memory Management**: Large datasets are processed in chunks with explicit garbage collection
- **Remote Usage**: Neuroglancer requires port forwarding when running on remote servers
- **Dependencies**: Install visualization packages only as needed to minimize setup complexity

---

## API Reference

For detailed function documentation, see the individual script files:

- **Data Conversion**: [`dfGen.py`](scripts/conversion/dfGen.py) - Extract and process vesicle mapping data
- **Mesh Generation**: [`neuron_mesh_gen.py`](scripts/conversion/neuron_mesh_gen.py), [`vesicle_mesh_gen.py`](scripts/conversion/vesicle_mesh_gen.py)
- **Visualization**: [`htmlGen.py`](scripts/visualizations/htmlGen.py), [`plotlyGen.py`](scripts/visualizations/plotlyGen.py), [`pyvistaGen.py`](scripts/visualizations/pyvistaGen.py), [`neuroglancerGen.py`](scripts/visualizations/neuroglancerGen.py)

Each script includes comprehensive help accessible via `python script_name.py --help`.