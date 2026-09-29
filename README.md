# genpark-inverted-file-ivf-centroid-clusterer-skill

Inverted File (IVF) index segmenting vector space into Voronoi cells for fast pruned vector search.

## Architecture

```mermaid
flowchart TD
    Query[Query Vector] --> FindCentroid[Locate Closest Centroid Cells]
    FindCentroid --> Scan[Scan Inverted Postings Lists for Selected Cells]
    Scan --> TopK[Exact Cosine Rank Top-K]
```

## Features
- **Configurable nprobe**: Trade-off speed versus recall accuracy.
- **Pure Python**: 100% standard library.
