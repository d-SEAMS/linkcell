//! N-dimensional exact neighbour search for feature-space workflows.
//!
//! The current tree is a vantage-point tree (VP-tree): exact metric search
//! with branch-and-bound pruning from the triangle inequality.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use crate::Error;

mod metric;
pub use metric::{Metric, PeriodicEuclidean};

/// One neighbour hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeighborHit {
    /// Point index in the indexed dataset.
    pub index: usize,
    /// Squared distance to the query.
    pub dist2: f64,
}

/// One undirected pair hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairHit {
    /// Lower point index.
    pub i: usize,
    /// Higher point index.
    pub j: usize,
    /// Squared distance.
    pub dist2: f64,
}

/// DBSCAN output.
#[derive(Clone, Debug, PartialEq)]
pub struct DbscanResult {
    /// Cluster labels, `-1` for noise.
    pub labels: Vec<i32>,
    /// `true` for core points.
    pub core: Vec<bool>,
    /// Number of discovered clusters.
    pub clusters: usize,
}

/// Coordinate storage for a metric index.
pub trait Space {
    /// Number of points.
    fn len(&self) -> usize;
    /// True when there are no points.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Point dimensionality.
    fn dims(&self) -> usize;
    /// Point coordinates.
    fn point(&self, index: usize) -> &[f64];
}

/// Public neighbour-index API.
pub trait NeighbourIndex {
    /// Number of indexed points.
    fn len(&self) -> usize;
    /// True when there are no indexed points.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Point dimensionality.
    fn dims(&self) -> usize;
    /// k-nearest neighbours, sorted by `(dist2, index)`.
    fn knn(&self, query: &[f64], k: usize) -> Result<Vec<NeighborHit>, Error>;
    /// Radius neighbours, sorted by `(dist2, index)`.
    fn within(&self, query: &[f64], radius: f64) -> Result<Vec<NeighborHit>, Error>;
    /// Radius neighbours of one indexed point, including itself at zero distance.
    fn within_index(&self, index: usize, radius: f64) -> Result<Vec<NeighborHit>, Error>;
    /// All index pairs `i < j` within `radius`.
    fn pairs_within(&self, radius: f64) -> Result<Vec<PairHit>, Error>;
}

/// Borrowed row-major point storage (`n * d` doubles).
#[derive(Clone, Copy)]
pub struct ContiguousSpace<'a> {
    data: &'a [f64],
    n: usize,
    d: usize,
}

/// Owned row-major point storage (`n * d` doubles).
#[derive(Clone)]
pub struct OwnedContiguousSpace {
    data: Vec<f64>,
    n: usize,
    d: usize,
}

impl<'a> ContiguousSpace<'a> {
    /// Build a borrowed view over `data` with shape `(n, d)`.
    pub fn from_row_major(data: &'a [f64], n: usize, d: usize) -> Result<Self, Error> {
        if d == 0 || n.checked_mul(d) != Some(data.len()) {
            return Err(Error::BadDimensions);
        }
        Ok(Self { data, n, d })
    }
}

impl Space for ContiguousSpace<'_> {
    fn len(&self) -> usize {
        self.n
    }

    fn dims(&self) -> usize {
        self.d
    }

    fn point(&self, index: usize) -> &[f64] {
        let off = index * self.d;
        &self.data[off..off + self.d]
    }
}

impl OwnedContiguousSpace {
    /// Build an owning space by copying `data`.
    pub fn from_row_major(data: &[f64], n: usize, d: usize) -> Result<Self, Error> {
        if d == 0 || n.checked_mul(d) != Some(data.len()) {
            return Err(Error::BadDimensions);
        }
        Ok(Self {
            data: data.to_vec(),
            n,
            d,
        })
    }
}

impl Space for OwnedContiguousSpace {
    fn len(&self) -> usize {
        self.n
    }

    fn dims(&self) -> usize {
        self.d
    }

    fn point(&self, index: usize) -> &[f64] {
        let off = index * self.d;
        &self.data[off..off + self.d]
    }
}

#[derive(Clone, Debug)]
struct Node {
    vp: usize,
    threshold2: f64,
    left: Option<usize>,
    right: Option<usize>,
}

/// Exact VP-tree index.
pub struct VpTreeIndex<S, M> {
    space: S,
    metric: M,
    nodes: Vec<Node>,
    root: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HeapEntry {
    dist2: f64,
    index: usize,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.dist2.total_cmp(&other.dist2) {
            Ordering::Equal => self.index.cmp(&other.index),
            ord => ord,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: Space + Sync, M: Metric + Sync> VpTreeIndex<S, M> {
    /// Build an index over `space` with `metric`.
    pub fn new(space: S, metric: M) -> Result<Self, Error> {
        if space.dims() == 0 || space.dims() != metric.dims() {
            return Err(Error::BadDimensions);
        }
        let mut me = Self {
            space,
            metric,
            nodes: Vec::new(),
            root: None,
        };
        let mut ids: Vec<usize> = (0..me.space.len()).collect();
        me.root = me.build_node(&mut ids)?;
        Ok(me)
    }

    fn build_node(&mut self, ids: &mut [usize]) -> Result<Option<usize>, Error> {
        if ids.is_empty() {
            return Ok(None);
        }
        if ids.len() == 1 {
            let id = self.nodes.len();
            self.nodes.push(Node {
                vp: ids[0],
                threshold2: 0.0,
                left: None,
                right: None,
            });
            return Ok(Some(id));
        }
        let vp = ids[ids.len() - 1];
        let anchor = self.space.point(vp);
        let mut dist: Vec<(usize, f64)> = ids[..ids.len() - 1]
            .iter()
            .map(|&idx| {
                self.metric
                    .dist2(anchor, self.space.point(idx))
                    .map(|d| (idx, d))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        dist.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let mid = dist.len() / 2;
        let threshold2 = dist[mid].1;
        let mut left_ids = Vec::with_capacity(mid);
        let mut right_ids = Vec::with_capacity(dist.len() - mid);
        for (pos, (idx, _)) in dist.into_iter().enumerate() {
            if pos < mid {
                left_ids.push(idx);
            } else {
                right_ids.push(idx);
            }
        }
        let node_id = self.nodes.len();
        self.nodes.push(Node {
            vp,
            threshold2,
            left: None,
            right: None,
        });
        let left = self.build_node(&mut left_ids)?;
        let right = self.build_node(&mut right_ids)?;
        self.nodes[node_id].left = left;
        self.nodes[node_id].right = right;
        Ok(Some(node_id))
    }

    fn check_query(&self, query: &[f64]) -> Result<(), Error> {
        if query.len() != self.space.dims() {
            return Err(Error::BadDimensions);
        }
        if !query.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(())
    }

    fn maybe_push(heap: &mut BinaryHeap<HeapEntry>, k: usize, hit: HeapEntry) {
        if k == 0 {
            return;
        }
        if heap.len() < k {
            heap.push(hit);
            return;
        }
        let worst = *heap.peek().expect("non-empty");
        if hit.dist2 < worst.dist2 || (hit.dist2 == worst.dist2 && hit.index < worst.index) {
            let _ = heap.pop();
            heap.push(hit);
        }
    }

    fn knn_rec(
        &self,
        node_id: Option<usize>,
        query: &[f64],
        k: usize,
        skip: Option<usize>,
        heap: &mut BinaryHeap<HeapEntry>,
    ) -> Result<(), Error> {
        let Some(nid) = node_id else {
            return Ok(());
        };
        let node = &self.nodes[nid];
        let idx = node.vp;
        let d2 = self.metric.dist2(query, self.space.point(idx))?;
        if skip != Some(idx) {
            Self::maybe_push(
                heap,
                k,
                HeapEntry {
                    dist2: d2,
                    index: idx,
                },
            );
        }
        if node.left.is_none() && node.right.is_none() {
            return Ok(());
        }
        let tau = if heap.len() < k {
            f64::INFINITY
        } else {
            heap.peek().map(|h| h.dist2.sqrt()).unwrap_or(f64::INFINITY)
        };
        let d = d2.sqrt();
        let mu = node.threshold2.sqrt();
        if d < mu {
            self.knn_rec(node.left, query, k, skip, heap)?;
            if d + tau >= mu {
                self.knn_rec(node.right, query, k, skip, heap)?;
            }
        } else {
            self.knn_rec(node.right, query, k, skip, heap)?;
            if d <= mu + tau {
                self.knn_rec(node.left, query, k, skip, heap)?;
            }
        }
        Ok(())
    }

    fn within_rec(
        &self,
        node_id: Option<usize>,
        query: &[f64],
        radius: f64,
        skip: Option<usize>,
        out: &mut Vec<NeighborHit>,
    ) -> Result<(), Error> {
        let Some(nid) = node_id else {
            return Ok(());
        };
        let node = &self.nodes[nid];
        let idx = node.vp;
        let d2 = self.metric.dist2(query, self.space.point(idx))?;
        if d2 <= radius * radius && skip != Some(idx) {
            out.push(NeighborHit {
                index: idx,
                dist2: d2,
            });
        }
        if node.left.is_none() && node.right.is_none() {
            return Ok(());
        }
        let d = d2.sqrt();
        let mu = node.threshold2.sqrt();
        if d - radius <= mu {
            self.within_rec(node.left, query, radius, skip, out)?;
        }
        if d + radius >= mu {
            self.within_rec(node.right, query, radius, skip, out)?;
        }
        Ok(())
    }

    fn knn_skip(
        &self,
        query: &[f64],
        k: usize,
        skip: Option<usize>,
    ) -> Result<Vec<NeighborHit>, Error> {
        let mut heap = BinaryHeap::new();
        self.knn_rec(self.root, query, k, skip, &mut heap)?;
        let mut out: Vec<_> = heap
            .into_iter()
            .map(|h| NeighborHit {
                index: h.index,
                dist2: h.dist2,
            })
            .collect();
        out.sort_by(|a, b| {
            a.dist2
                .total_cmp(&b.dist2)
                .then_with(|| a.index.cmp(&b.index))
        });
        Ok(out)
    }

    fn within_skip(
        &self,
        query: &[f64],
        radius: f64,
        skip: Option<usize>,
    ) -> Result<Vec<NeighborHit>, Error> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(Error::BadCutoff);
        }
        let mut out = Vec::new();
        self.within_rec(self.root, query, radius, skip, &mut out)?;
        out.sort_by(|a, b| {
            a.dist2
                .total_cmp(&b.dist2)
                .then_with(|| a.index.cmp(&b.index))
        });
        Ok(out)
    }

    /// Batched k-nearest queries from one packed row-major query buffer.
    #[cfg(feature = "parallel")]
    pub fn knn_many(&self, queries: &[f64], k: usize) -> Result<Vec<Vec<NeighborHit>>, Error> {
        use rayon::prelude::*;
        if self.space.dims() == 0 || queries.len() % self.space.dims() != 0 {
            return Err(Error::BadDimensions);
        }
        let d = self.space.dims();
        queries
            .par_chunks(d)
            .map(|q| {
                self.check_query(q)?;
                self.knn_skip(q, k, None)
            })
            .collect::<Result<Vec<_>, Error>>()
    }

    /// Serial batched k-nearest queries, for no-`parallel` builds.
    #[cfg(not(feature = "parallel"))]
    pub fn knn_many(&self, queries: &[f64], k: usize) -> Result<Vec<Vec<NeighborHit>>, Error> {
        if self.space.dims() == 0 || queries.len() % self.space.dims() != 0 {
            return Err(Error::BadDimensions);
        }
        let d = self.space.dims();
        let mut out = Vec::with_capacity(queries.len() / d);
        for q in queries.chunks(d) {
            self.check_query(q)?;
            out.push(self.knn_skip(q, k, None)?);
        }
        Ok(out)
    }

    /// Batched radius queries from one packed row-major query buffer.
    #[cfg(feature = "parallel")]
    pub fn within_many(
        &self,
        queries: &[f64],
        radius: f64,
    ) -> Result<Vec<Vec<NeighborHit>>, Error> {
        use rayon::prelude::*;
        if self.space.dims() == 0 || queries.len() % self.space.dims() != 0 {
            return Err(Error::BadDimensions);
        }
        let d = self.space.dims();
        queries
            .par_chunks(d)
            .map(|q| {
                self.check_query(q)?;
                self.within_skip(q, radius, None)
            })
            .collect::<Result<Vec<_>, Error>>()
    }

    /// Serial batched radius queries, for no-`parallel` builds.
    #[cfg(not(feature = "parallel"))]
    pub fn within_many(
        &self,
        queries: &[f64],
        radius: f64,
    ) -> Result<Vec<Vec<NeighborHit>>, Error> {
        if self.space.dims() == 0 || queries.len() % self.space.dims() != 0 {
            return Err(Error::BadDimensions);
        }
        let d = self.space.dims();
        let mut out = Vec::with_capacity(queries.len() / d);
        for q in queries.chunks(d) {
            self.check_query(q)?;
            out.push(self.within_skip(q, radius, None)?);
        }
        Ok(out)
    }

    /// Deterministic DBSCAN (`min_pts` includes the point itself).
    ///
    /// Border points keep the first cluster that reaches them while seeds are
    /// scanned in ascending index order and each core expansion visits
    /// neighbours in ascending index order.
    pub fn dbscan(&self, eps: f64, min_pts: usize) -> Result<DbscanResult, Error> {
        if !eps.is_finite() || eps <= 0.0 || min_pts == 0 {
            return Err(Error::BadCutoff);
        }
        let n = self.space.len();
        let mut labels = vec![-2; n];
        let mut core = vec![false; n];
        let mut cluster = 0i32;
        const UNASSIGNED: i32 = -2;
        const NOISE: i32 = -1;
        for i in 0..n {
            if labels[i] != UNASSIGNED {
                continue;
            }
            let mut neigh = self.within_index(i, eps)?;
            neigh.sort_by_key(|h| h.index);
            if neigh.len() < min_pts {
                labels[i] = NOISE;
                continue;
            }
            labels[i] = cluster;
            core[i] = true;
            let mut queued = vec![false; n];
            let mut queue = VecDeque::new();
            for h in neigh {
                if h.index != i && !queued[h.index] {
                    queued[h.index] = true;
                    queue.push_back(h.index);
                }
            }
            while let Some(p) = queue.pop_front() {
                if labels[p] == NOISE {
                    labels[p] = cluster;
                }
                if labels[p] != UNASSIGNED {
                    continue;
                }
                labels[p] = cluster;
                let mut p_neigh = self.within_index(p, eps)?;
                p_neigh.sort_by_key(|h| h.index);
                if p_neigh.len() >= min_pts {
                    core[p] = true;
                    for h in p_neigh {
                        if !queued[h.index] {
                            queued[h.index] = true;
                            queue.push_back(h.index);
                        }
                    }
                }
            }
            cluster += 1;
        }
        Ok(DbscanResult {
            labels,
            core,
            clusters: cluster as usize,
        })
    }
}

impl<S: Space + Sync, M: Metric + Sync> NeighbourIndex for VpTreeIndex<S, M> {
    fn len(&self) -> usize {
        self.space.len()
    }

    fn dims(&self) -> usize {
        self.space.dims()
    }

    fn knn(&self, query: &[f64], k: usize) -> Result<Vec<NeighborHit>, Error> {
        self.check_query(query)?;
        self.knn_skip(query, k, None)
    }

    fn within(&self, query: &[f64], radius: f64) -> Result<Vec<NeighborHit>, Error> {
        self.check_query(query)?;
        self.within_skip(query, radius, None)
    }

    fn within_index(&self, index: usize, radius: f64) -> Result<Vec<NeighborHit>, Error> {
        if index >= self.space.len() {
            return Err(Error::Overflow);
        }
        self.within_skip(self.space.point(index), radius, None)
    }

    fn pairs_within(&self, radius: f64) -> Result<Vec<PairHit>, Error> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(Error::BadCutoff);
        }
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let nested: Vec<Vec<PairHit>> = (0..self.space.len())
                .into_par_iter()
                .map(|i| -> Result<Vec<PairHit>, Error> {
                    let hits = self.within_skip(self.space.point(i), radius, Some(i))?;
                    Ok(hits
                        .into_iter()
                        .filter(|h| i < h.index)
                        .map(|h| PairHit {
                            i,
                            j: h.index,
                            dist2: h.dist2,
                        })
                        .collect())
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(nested.into_iter().flatten().collect())
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut out = Vec::new();
            for i in 0..self.space.len() {
                let hits = self.within_skip(self.space.point(i), radius, Some(i))?;
                out.extend(hits.into_iter().filter(|h| i < h.index).map(|h| PairHit {
                    i,
                    j: h.index,
                    dist2: h.dist2,
                }));
            }
            Ok(out)
        }
    }
}
