//! N-dimensional exact neighbour search for feature-space workflows.
//!
//! The current tree is a vantage-point tree (VP-tree): exact metric search
//! with branch-and-bound pruning from the triangle inequality.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use crate::Error;

mod metric;
pub use metric::{Euclidean, Metric, PeriodicEuclidean, PeriodicSpace};

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
///
/// Stored points and each query are reduced with [`Metric::write_canonical`]
/// once. Pruning widens the triangle test by a relative Higham factor and by
/// [`Metric::vp_absolute_slack`].
pub struct VpTreeIndex<S, M> {
    space: S,
    metric: M,
    nodes: Vec<Node>,
    root: Option<usize>,
    /// Row-major representatives. Periodic axes lie in `[0, L)`.
    canonical: Vec<f64>,
    /// Absolute triangle slack for this metric. Periodic residue error plus
    /// the square-underflow floor, both proved upper bounds.
    absolute_slack: f64,
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
    fn gamma(n: usize) -> f64 {
        // Higham: gamma_n = n*u / (1 - n*u), where u is unit roundoff.
        let u = 0.5 * f64::EPSILON;
        let nu = (n as f64) * u;
        if nu >= 1.0 {
            return f64::INFINITY;
        }
        nu / (1.0 - nu)
    }

    fn pruning_margin(&self, a: f64, b: f64, c: f64) -> f64 {
        // Relative guard for the norm, the square root, and the branch test.
        // The absolute term is Metric::vp_absolute_slack: the square-underflow
        // floor 8 * sqrt(dims) * 2^-537, plus the periodic residue slack when
        // an axis has a period. The floor stays positive for every non-zero
        // dimension, including an open axis.
        let d = self.space.dims();
        let n_ops = d.saturating_mul(6).saturating_add(16);
        let g_dist2 = Self::gamma(n_ops);
        let g_sqrt = Self::gamma(2);
        let g_add = Self::gamma(3);
        if !g_dist2.is_finite() || !g_sqrt.is_finite() || !g_add.is_finite() {
            return f64::INFINITY;
        }
        let rel = (1.0 + g_dist2) * (1.0 + g_sqrt) * (1.0 + g_add) - 1.0;
        let scale = a.abs() + b.abs() + c.abs();
        let relative = rel * scale + 16.0 * (Self::ulp(a) + Self::ulp(b) + Self::ulp(c));
        let margin = relative + self.absolute_slack;
        if margin.is_finite() && margin >= 0.0 {
            margin
        } else {
            f64::INFINITY
        }
    }

    fn visit_left(d: f64, mu: f64, tau: f64, margin: f64) -> bool {
        // Left holds dist(vp, p) <= mu. A point there satisfies
        // dist(q, p) >= d - mu - defect, so it can be inside tau only when
        // d <= mu + tau + defect.
        d <= mu + tau + margin
    }

    fn visit_right(d: f64, mu: f64, tau: f64, margin: f64) -> bool {
        // Right holds dist(vp, p) >= mu, so dist(q, p) >= mu - d - defect.
        d + tau + margin >= mu
    }

    fn point_canon(&self, index: usize) -> &[f64] {
        let d = self.space.dims();
        let off = index * d;
        &self.canonical[off..off + d]
    }

    fn ulp(x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        let next = if x >= 0.0 {
            f64::from_bits(x.to_bits().saturating_add(1))
        } else {
            f64::from_bits(x.to_bits().saturating_sub(1))
        };
        (next - x).abs()
    }

    /// Build an index over `space` with `metric`.
    pub fn new(space: S, metric: M) -> Result<Self, Error> {
        if space.dims() == 0 || space.dims() != metric.dims() {
            return Err(Error::BadDimensions);
        }
        let n = space.len();
        let d = space.dims();
        for i in 0..n {
            let p = space.point(i);
            if p.len() != d {
                return Err(Error::BadDimensions);
            }
            if !p.iter().copied().all(f64::is_finite) {
                return Err(Error::NonFinite);
            }
        }
        let slack = metric.vp_absolute_slack();
        let absolute_slack = if slack.is_finite() && slack >= 0.0 {
            slack
        } else {
            f64::INFINITY
        };
        let Some(canonical_len) = n.checked_mul(d) else {
            return Err(Error::Overflow);
        };
        let mut canonical = vec![0.0; canonical_len];
        for i in 0..n {
            let dst = &mut canonical[i * d..(i + 1) * d];
            metric.write_canonical(space.point(i), dst)?;
        }
        let mut me = Self {
            space,
            metric,
            nodes: Vec::new(),
            root: None,
            canonical,
            absolute_slack,
        };
        me.root = me.build_iter()?;
        Ok(me)
    }

    /// One node per point. The explicit stack is on the heap, so a duplicate
    /// cloud cannot overflow the calling thread. Ties split by count: the
    /// left half has `dist <= threshold` and the right half has
    /// `dist >= threshold`, and both halves are non-empty whenever more than
    /// one point remains beside the vantage point.
    fn build_iter(&mut self) -> Result<Option<usize>, Error> {
        let n = self.space.len();
        if n == 0 {
            return Ok(None);
        }
        let mut ids: Vec<usize> = (0..n).collect();
        struct Job {
            start: usize,
            end: usize,
            parent: Option<usize>,
            is_right: bool,
        }
        let mut stack = Vec::new();
        stack.push(Job {
            start: 0,
            end: n,
            parent: None,
            is_right: false,
        });
        let mut root = None;
        self.nodes.reserve(n);
        while let Some(job) = stack.pop() {
            let len = job.end - job.start;
            if len == 0 {
                continue;
            }
            let node_id = self.nodes.len();
            if let Some(parent) = job.parent {
                if job.is_right {
                    self.nodes[parent].right = Some(node_id);
                } else {
                    self.nodes[parent].left = Some(node_id);
                }
            } else {
                root = Some(node_id);
            }
            if len == 1 {
                self.nodes.push(Node {
                    vp: ids[job.start],
                    threshold2: 0.0,
                    left: None,
                    right: None,
                });
                continue;
            }
            let vp = ids[job.end - 1];
            let anchor = self.point_canon(vp).to_vec();
            let count = len - 1;
            let mut dist = Vec::with_capacity(count);
            for &idx in &ids[job.start..job.end - 1] {
                let d2 = self.metric.dist2(&anchor, self.point_canon(idx))?;
                dist.push((idx, d2));
            }
            dist.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            if count == 1 {
                let threshold2 = dist[0].1;
                ids[job.start] = dist[0].0;
                self.nodes.push(Node {
                    vp,
                    threshold2,
                    left: None,
                    right: None,
                });
                stack.push(Job {
                    start: job.start,
                    end: job.start + 1,
                    parent: Some(node_id),
                    is_right: true,
                });
                continue;
            }
            let mid = count / 2;
            let threshold2 = dist[mid].1;
            debug_assert!(dist[..mid].iter().all(|p| p.1 <= threshold2));
            debug_assert!(dist[mid..].iter().all(|p| p.1 >= threshold2));
            for (i, (idx, _)) in dist.iter().enumerate() {
                ids[job.start + i] = *idx;
            }
            self.nodes.push(Node {
                vp,
                threshold2,
                left: None,
                right: None,
            });
            stack.push(Job {
                start: job.start + mid,
                end: job.end - 1,
                parent: Some(node_id),
                is_right: true,
            });
            stack.push(Job {
                start: job.start,
                end: job.start + mid,
                parent: Some(node_id),
                is_right: false,
            });
        }
        Ok(root)
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

    fn tau_of(heap: &BinaryHeap<HeapEntry>, k: usize) -> f64 {
        if heap.len() < k {
            f64::INFINITY
        } else {
            heap.peek().map(|h| h.dist2.sqrt()).unwrap_or(f64::INFINITY)
        }
    }

    fn knn_walk(
        &self,
        query: &[f64],
        k: usize,
        skip: Option<usize>,
        heap: &mut BinaryHeap<HeapEntry>,
    ) -> Result<(), Error> {
        let Some(root) = self.root else {
            return Ok(());
        };
        struct Frame {
            nid: usize,
            phase: u8,
            d: f64,
            mu: f64,
            far: Option<usize>,
            far_is_right: bool,
        }
        let mut stack = Vec::new();
        stack.push(Frame {
            nid: root,
            phase: 0,
            d: 0.0,
            mu: 0.0,
            far: None,
            far_is_right: false,
        });
        while let Some(frame) = stack.pop() {
            if frame.phase == 1 {
                let tau = Self::tau_of(heap, k);
                let margin = self.pruning_margin(frame.d, tau, frame.mu);
                let go = if frame.far_is_right {
                    Self::visit_right(frame.d, frame.mu, tau, margin)
                } else {
                    Self::visit_left(frame.d, frame.mu, tau, margin)
                };
                if go {
                    if let Some(fid) = frame.far {
                        stack.push(Frame {
                            nid: fid,
                            phase: 0,
                            d: 0.0,
                            mu: 0.0,
                            far: None,
                            far_is_right: false,
                        });
                    }
                }
                continue;
            }
            let (vp, left, right, threshold2) = {
                let node = &self.nodes[frame.nid];
                (node.vp, node.left, node.right, node.threshold2)
            };
            let d2 = self.metric.dist2(query, self.point_canon(vp))?;
            if skip != Some(vp) {
                Self::maybe_push(
                    heap,
                    k,
                    HeapEntry {
                        dist2: d2,
                        index: vp,
                    },
                );
            }
            if left.is_none() && right.is_none() {
                continue;
            }
            let d = d2.sqrt();
            let mu = threshold2.sqrt();
            let tau = Self::tau_of(heap, k);
            let margin = self.pruning_margin(d, tau, mu);
            let prefer_left = d < mu;
            let (near, far, far_is_right) = if prefer_left {
                (left, right, true)
            } else {
                (right, left, false)
            };
            let visit_near = if prefer_left {
                Self::visit_left(d, mu, tau, margin)
            } else {
                Self::visit_right(d, mu, tau, margin)
            };
            let visit_far = if far_is_right {
                Self::visit_right(d, mu, tau, margin)
            } else {
                Self::visit_left(d, mu, tau, margin)
            };
            if let Some(near_id) = near {
                if visit_near {
                    stack.push(Frame {
                        nid: frame.nid,
                        phase: 1,
                        d,
                        mu,
                        far,
                        far_is_right,
                    });
                    stack.push(Frame {
                        nid: near_id,
                        phase: 0,
                        d: 0.0,
                        mu: 0.0,
                        far: None,
                        far_is_right: false,
                    });
                    continue;
                }
            }
            if visit_far {
                if let Some(far_id) = far {
                    stack.push(Frame {
                        nid: far_id,
                        phase: 0,
                        d: 0.0,
                        mu: 0.0,
                        far: None,
                        far_is_right: false,
                    });
                }
            }
        }
        Ok(())
    }

    fn within_walk(
        &self,
        query: &[f64],
        radius: f64,
        skip: Option<usize>,
        out: &mut Vec<NeighborHit>,
    ) -> Result<(), Error> {
        let Some(root) = self.root else {
            return Ok(());
        };
        let mut stack = Vec::new();
        stack.push(root);
        let r2 = radius * radius;
        while let Some(nid) = stack.pop() {
            let (vp, left, right, threshold2) = {
                let node = &self.nodes[nid];
                (node.vp, node.left, node.right, node.threshold2)
            };
            let d2 = self.metric.dist2(query, self.point_canon(vp))?;
            if d2 <= r2 && skip != Some(vp) {
                out.push(NeighborHit {
                    index: vp,
                    dist2: d2,
                });
            }
            if left.is_none() && right.is_none() {
                continue;
            }
            let d = d2.sqrt();
            let mu = threshold2.sqrt();
            let margin = self.pruning_margin(d, radius, mu);
            if let Some(id) = left {
                if Self::visit_left(d, mu, radius, margin) {
                    stack.push(id);
                }
            }
            if let Some(id) = right {
                if Self::visit_right(d, mu, radius, margin) {
                    stack.push(id);
                }
            }
        }
        Ok(())
    }

    fn prepare_query(&self, query: &[f64]) -> Result<Vec<f64>, Error> {
        let mut prepared = vec![0.0; self.space.dims()];
        self.metric.write_canonical(query, &mut prepared)?;
        Ok(prepared)
    }

    fn knn_skip(
        &self,
        query: &[f64],
        k: usize,
        skip: Option<usize>,
    ) -> Result<Vec<NeighborHit>, Error> {
        let prepared = self.prepare_query(query)?;
        let mut heap = BinaryHeap::new();
        self.knn_walk(&prepared, k, skip, &mut heap)?;
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
            return Err(Error::BadNdRadius);
        }
        let prepared = self.prepare_query(query)?;
        let mut out = Vec::new();
        self.within_walk(&prepared, radius, skip, &mut out)?;
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
            return Err(Error::BadNdEps);
        }
        let n = self.space.len();
        let mut labels = vec![-2; n];
        let mut core = vec![false; n];
        let mut queued = vec![0u32; n];
        let mut stamp = 1u32;
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
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                queued.fill(0);
                stamp = 1;
            }
            let mut queue = VecDeque::new();
            for h in neigh {
                if h.index != i && queued[h.index] != stamp {
                    queued[h.index] = stamp;
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
                        if queued[h.index] != stamp {
                            queued[h.index] = stamp;
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
            return Err(Error::IndexOutOfRange);
        }
        self.within_skip(self.space.point(index), radius, None)
    }

    fn pairs_within(&self, radius: f64) -> Result<Vec<PairHit>, Error> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(Error::BadNdRadius);
        }
        #[cfg(feature = "parallel")]
        {
            if crate::on_caller() {
                let mut out = Vec::new();
                for i in 0..self.space.len() {
                    let hits = self.within_skip(self.space.point(i), radius, Some(i))?;
                    out.extend(hits.into_iter().filter(|h| i < h.index).map(|h| PairHit {
                        i,
                        j: h.index,
                        dist2: h.dist2,
                    }));
                }
                out.sort_by(|a, b| a.i.cmp(&b.i).then_with(|| a.j.cmp(&b.j)));
                Ok(out)
            } else {
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
                let mut out: Vec<_> = nested.into_iter().flatten().collect();
                out.sort_by(|a, b| a.i.cmp(&b.i).then_with(|| a.j.cmp(&b.j)));
                Ok(out)
            }
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
            out.sort_by(|a, b| a.i.cmp(&b.i).then_with(|| a.j.cmp(&b.j)));
            Ok(out)
        }
    }
}
