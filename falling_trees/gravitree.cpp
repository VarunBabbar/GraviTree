#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef FALLING_TREES_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#endif

using std::size_t;

// ------------------------ Tree Structures ------------------------
struct Tree;
using TreePtr = std::shared_ptr<Tree>;

struct Tree {
  bool is_leaf = false;
  // Leaf fields
  double pred_prob = 0.0;
  // Node fields
  int feature = -1;
  TreePtr left;
  TreePtr right;
  // Shared field
  double objective = 0.0;

  // Cached hash for structural hashing.
  mutable bool hash_cached = false;
  mutable size_t hash_cache = 0;
};

static inline TreePtr make_leaf(double pred_prob, double objective) {
  auto t = std::make_shared<Tree>();
  t->is_leaf = true;
  t->pred_prob = pred_prob;
  t->objective = objective;
  return t;
}

static inline TreePtr make_node(int feature, const TreePtr& left, const TreePtr& right, double objective) {
  auto t = std::make_shared<Tree>();
  t->is_leaf = false;
  t->feature = feature;
  t->left = left;
  t->right = right;
  t->objective = objective;
  return t;
}

// ------------------------ Path Key ------------------------
// Inline-storage sorted-unique sequence of (feature, direction) pairs.
// kInlineCapacity bounds the path depth; with depth <= 5 in current
// experiments, 16 leaves comfortable headroom. Avoiding heap allocation
// here is a measurable win because PathKey is copied on every recursive
// call into OptFallingTree/OptFallingRset.
struct PathKey {
  static constexpr uint8_t kInlineCapacity = 16;
  std::array<std::pair<int, char>, kInlineCapacity> pairs{};
  uint8_t size_ = 0;

  uint8_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  const std::pair<int, char>* begin() const noexcept { return pairs.data(); }
  const std::pair<int, char>* end() const noexcept { return pairs.data() + size_; }
};

static inline PathKey normalize_path(const std::vector<std::pair<int, char>>& pairs) {
  std::vector<std::pair<int, char>> tmp(pairs);
  std::sort(tmp.begin(), tmp.end());
  tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
  PathKey key;
  assert(tmp.size() <= PathKey::kInlineCapacity);
  for (size_t i = 0; i < tmp.size(); ++i) {
    key.pairs[i] = tmp[i];
  }
  key.size_ = static_cast<uint8_t>(tmp.size());
  return key;
}

// Insert (feature, direction) maintaining the sorted-unique invariant.
// O(d) shift; no heap allocation.
static inline PathKey path_add_pair(const PathKey& path, int feature, char direction) {
  PathKey next = path;
  const std::pair<int, char> v{feature, direction};
  uint8_t i = 0;
  while (i < next.size_ && next.pairs[i] < v) {
    ++i;
  }
  if (i < next.size_ && next.pairs[i] == v) {
    return next;
  }
  assert(next.size_ < PathKey::kInlineCapacity);
  for (uint8_t k = next.size_; k > i; --k) {
    next.pairs[k] = next.pairs[k - 1];
  }
  next.pairs[i] = v;
  ++next.size_;
  return next;
}

// Remove (feature, direction). Output stays sorted because input was sorted.
static inline PathKey path_remove_pair(const PathKey& path, int feature, char direction) {
  PathKey next;
  const std::pair<int, char> v{feature, direction};
  for (uint8_t i = 0; i < path.size_; ++i) {
    if (path.pairs[i] == v) {
      continue;
    }
    next.pairs[next.size_++] = path.pairs[i];
  }
  return next;
}

struct PathKeyHash {
  size_t operator()(const PathKey& key) const noexcept {
    size_t h = 1469598103934665603ULL;
    for (uint8_t i = 0; i < key.size_; ++i) {
      const uint64_t v =
          (static_cast<uint64_t>(static_cast<uint32_t>(key.pairs[i].first)) << 8) ^
          static_cast<uint64_t>(static_cast<unsigned char>(key.pairs[i].second));
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
  }
};

struct PathKeyEq {
  bool operator()(const PathKey& a, const PathKey& b) const noexcept {
    if (a.size_ != b.size_) return false;
    for (uint8_t i = 0; i < a.size_; ++i) {
      if (a.pairs[i] != b.pairs[i]) return false;
    }
    return true;
  }
};

static inline PathKey path_to_key(const std::string& path) {
  if (path.empty()) {
    return PathKey{};
  }
  std::string normalized = path;
  if (!normalized.empty() && normalized[0] == '|') {
    normalized.erase(0, 1);
  }
  if (normalized.empty()) {
    return PathKey{};
  }
  std::vector<std::pair<int, char>> pairs;
  std::stringstream ss(normalized);
  std::string token;
  while (std::getline(ss, token, '|')) {
    auto pos = token.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    std::string feature_str = token.substr(0, pos);
    std::string dir_str = token.substr(pos + 1);
    try {
      int feature = std::stoi(feature_str);
      if (!dir_str.empty()) {
        pairs.emplace_back(feature, dir_str[0]);
      }
    } catch (...) {
      continue;
    }
  }
  return normalize_path(pairs);
}

static inline std::string key_to_path(const PathKey& path) {
  if (path.empty()) {
    return "";
  }
  std::ostringstream oss;
  for (uint8_t i = 0; i < path.size(); ++i) {
    if (i > 0) {
      oss << "|";
    }
    oss << path.pairs[i].first << ":" << path.pairs[i].second;
  }
  return oss.str();
}

// ------------------------ Solution / OptResult ------------------------
// A single (loss, tree, max-positive-prob) solution triple S = (L, T, p^+).
struct Solution {
  double loss;
  TreePtr tree;
  double pmax;
};

// Result of OptFallingTree: best solution S* (possibly a leaf) and best
// non-leaf (internal-rooted) solution S^diamond (nullopt if none exists).
struct OptResult {
  Solution best;
  std::optional<Solution> best_internal;
};

// ------------------------ Subproblem Cache ------------------------
static double g_budget_bucket_size = 1e-3;
static inline int budget_bucket(double v) {
  if (v < 0.0) {
    v = 0.0;
  } else if (v > 1.0) {
    v = 1.0;
  }
  const double bucket_size = g_budget_bucket_size > 0.0 ? g_budget_bucket_size : 1e-3;
  return static_cast<int>(std::llround(v / bucket_size));
}

struct CacheEntry {
  double best_loss = 0.0;
  double pmax_best = 0.0;
  bool is_root_internal = false;
  std::optional<double> current_leaf_prob_on_path;
  // Cached subproblem solution. Populated by writers in OptFallingTree /
  // OptFallingRset so that OptFallingTree can short-circuit at function
  // entry. A null `tree` means the entry only carries summary fields
  // (e.g. legacy writers) and is not usable for entry memoization.
  TreePtr tree;
  std::optional<Solution> best_internal;
};

struct CacheKey {
  PathKey path;
  int depth = 0;
  int budget_bucket = 0;
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& k) const noexcept {
    size_t h1 = PathKeyHash{}(k.path);
    size_t h2 = std::hash<int>{}(k.depth);
    size_t h3 = std::hash<int>{}(k.budget_bucket);
    size_t h = h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    return h ^ (h3 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
  }
};

struct CacheKeyEq {
  bool operator()(const CacheKey& a, const CacheKey& b) const noexcept {
    return a.depth == b.depth && a.budget_bucket == b.budget_bucket && PathKeyEq{}(a.path, b.path);
  }
};

static size_t MAX_CACHE_SIZE = 100000;
static size_t _cache_hits = 0;
static size_t _cache_misses = 0;

static std::unordered_map<CacheKey, CacheEntry, CacheKeyHash, CacheKeyEq>
    _subproblem_cache;

// Bulk half-evict on overflow. This amortizes to O(1) per insertion and
// removes the per-access list bookkeeping that a strict LRU policy would
// otherwise impose (every cache hit used to copy a PathKey into a fresh
// std::list node, which dominated the cache cost).
static inline void cache_set(const CacheKey& key, const CacheEntry& value) {
  if (_subproblem_cache.size() >= MAX_CACHE_SIZE) {
    const size_t target = MAX_CACHE_SIZE / 2;
    auto it = _subproblem_cache.begin();
    while (_subproblem_cache.size() > target && it != _subproblem_cache.end()) {
      it = _subproblem_cache.erase(it);
    }
  }
  _subproblem_cache[key] = value;
}

static inline std::optional<CacheEntry> cache_get(const CacheKey& key) {
  auto it = _subproblem_cache.find(key);
  if (it == _subproblem_cache.end()) {
    ++_cache_misses;
    return std::nullopt;
  }
  ++_cache_hits;
  return it->second;
}

static inline void cache_clear() {
  _subproblem_cache.clear();
  _cache_hits = 0;
  _cache_misses = 0;
}

// ------------------------ Tree Utility Functions ------------------------
static inline double pos_prop(const std::vector<int>& y, const std::vector<int>& row_idx) {
  if (row_idx.empty()) {
    return 0.0;
  }
  double pos = 0.0;
  for (int idx : row_idx) {
    pos += y[idx];
  }
  return pos / static_cast<double>(row_idx.size());
}

static inline std::pair<double, double> leaf_cost(
    const std::vector<int>& y, const std::vector<int>& row_idx, double lam, int n) {
  if (row_idx.empty()) {
    return {0.0, 0.0};
  }
  int pos = 0;
  for (int idx : row_idx) {
    pos += y[idx];
  }
  int neg = static_cast<int>(row_idx.size()) - pos;
  double loss = std::min(static_cast<double>(pos) / static_cast<double>(n),
                         static_cast<double>(neg) / static_cast<double>(n));
  loss += lam;
  double prob = static_cast<double>(pos) / static_cast<double>(row_idx.size());
  return {loss, prob};
}

static inline double tree_obj(const TreePtr& tree) {
  if (!tree) {
    return 0.0;
  }
  if (tree->is_leaf) {
    return tree->objective;
  }
  return tree_obj(tree->left) + tree_obj(tree->right);
}

static void print_tree(const TreePtr& t, int depth);

static size_t tree_hash(const TreePtr& tree) {
  if (!tree) {
    return 0x9e3779b97f4a7c15ULL;
  }
  if (tree->hash_cached) {
    return tree->hash_cache;
  }
  size_t h = 0;
  if (tree->is_leaf) {
    size_t h1 = std::hash<double>{}(tree->pred_prob);
    size_t h2 = std::hash<double>{}(tree->objective);
    h = h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  } else {
    size_t h1 = std::hash<int>{}(tree->feature);
    size_t h2 = tree_hash(tree->left);
    size_t h3 = tree_hash(tree->right);
    size_t h4 = std::hash<double>{}(tree->objective);
    h = h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    h ^= h3 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= h4 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  }
  tree->hash_cache = h;
  tree->hash_cached = true;
  return h;
}

static bool tree_equal(const TreePtr& a, const TreePtr& b) {
  if (a == b) {
    return true;
  }
  if (!a || !b) {
    return false;
  }
  if (a->is_leaf != b->is_leaf) {
    return false;
  }
  if (a->objective != b->objective) {
    return false;
  }
  if (a->is_leaf) {
    return a->pred_prob == b->pred_prob;
  }
  if (a->feature != b->feature) {
    return false;
  }
  return tree_equal(a->left, b->left) && tree_equal(a->right, b->right);
}

static inline double get_max_prob(const TreePtr& tree) {
  if (!tree) {
    return 0.0;
  }
  if (tree->is_leaf) {
    return tree->pred_prob;
  }
  return std::max(get_max_prob(tree->left), get_max_prob(tree->right));
}

static inline std::unordered_map<int, double> get_max_prob_at_every_level(const TreePtr& tree) {
  std::unordered_map<int, double> result;
  if (!tree) {
    return result;
  }
  if (tree->is_leaf) {
    result[0] = tree->pred_prob;
    return result;
  }
  auto left_map = get_max_prob_at_every_level(tree->left);
  auto right_map = get_max_prob_at_every_level(tree->right);
  int max_level = 0;
  for (const auto& kv : left_map) {
    max_level = std::max(max_level, kv.first);
  }
  for (const auto& kv : right_map) {
    max_level = std::max(max_level, kv.first);
  }
  for (int l = 0; l <= max_level; ++l) {
    double lv = left_map.count(l) ? left_map[l] : 0.0;
    double rv = right_map.count(l) ? right_map[l] : 0.0;
    result[l] = std::max(lv, rv);
  }
  return result;
}

static inline bool is_falling_tree(const TreePtr& tree) {
  if (!tree || tree->is_leaf) {
    return true;
  }
  if (!tree->left->is_leaf && !tree->right->is_leaf) {
    return is_falling_tree(tree->left) && is_falling_tree(tree->right);
  }
  if (!tree->left->is_leaf && tree->right->is_leaf) {
    double max_left_prob = get_max_prob(tree->left);
    if (tree->right->pred_prob < max_left_prob) {
      return false;
    }
  }
  if (tree->left->is_leaf && !tree->right->is_leaf) {
    double max_right_prob = get_max_prob(tree->right);
    if (tree->left->pred_prob < max_right_prob) {
      return false;
    }
  }
  return true;
}

static inline bool is_falling_tree_new_constraint(const TreePtr& tree) {
  std::function<bool(const TreePtr&, double)> check = [&](const TreePtr& node, double last_prob) -> bool {
    if (!node) {
      return true;
    }
    if (node->is_leaf) {
      return node->pred_prob <= last_prob;
    }
    const auto& left = node->left;
    const auto& right = node->right;
    if (left->is_leaf && right->is_leaf) {
      return std::max(left->pred_prob, right->pred_prob) <= last_prob;
    }
    if (left->is_leaf && !right->is_leaf) {
      if (left->pred_prob > last_prob) {
        return false;
      }
      return check(right, left->pred_prob);
    }
    if (right->is_leaf && !left->is_leaf) {
      if (right->pred_prob > last_prob) {
        return false;
      }
      return check(left, right->pred_prob);
    }
    return check(left, last_prob) && check(right, last_prob);
  };
  if (!tree) {
    return true;
  }
  return check(tree, std::numeric_limits<double>::infinity());
}

static inline int tree_height(const TreePtr& tree) {
  if (!tree) {
    return -1;
  }
  if (tree->is_leaf) {
    return 0;
  }
  return 1 + std::max(tree_height(tree->left), tree_height(tree->right));
}

static inline int count_leaves(const TreePtr& tree) {
  if (!tree) {
    return 0;
  }
  if (tree->is_leaf) {
    return 1;
  }
  return count_leaves(tree->left) + count_leaves(tree->right);
}

static inline int colless_index(const TreePtr& tree) {
  if (!tree || tree->is_leaf) {
    return 0;
  }
  int left_leaves = count_leaves(tree->left);
  int right_leaves = count_leaves(tree->right);
  int current_diff = std::abs(left_leaves - right_leaves);
  return current_diff + colless_index(tree->left) + colless_index(tree->right);
}

static inline double normalized_colless_index(const TreePtr& tree) {
  int n = count_leaves(tree);
  if (n <= 1) {
    return 1.0;
  }
  int raw = colless_index(tree);
  int max_colless = (n - 1) * (n - 2) / 2;
  if (max_colless == 0) {
    return 1.0;
  }
  return static_cast<double>(raw) / static_cast<double>(max_colless);
}

// ------------------------ Cache Helpers ------------------------
static inline CacheEntry make_cache_entry(
    const TreePtr& tree,
    double loss,
    std::optional<double> pmax,
    int depth,
    std::optional<double> current_leaf_prob_on_path,
    std::optional<Solution> best_internal = std::nullopt) {
  if (!pmax.has_value()) {
    pmax = tree->is_leaf ? tree->pred_prob : get_max_prob(tree);
  }
  bool is_internal = (!tree->is_leaf && !tree->left->is_leaf && !tree->right->is_leaf);
  CacheEntry entry;
  entry.best_loss = loss;
  entry.pmax_best = pmax.value();
  entry.is_root_internal = is_internal;
  entry.current_leaf_prob_on_path = current_leaf_prob_on_path;
  entry.tree = tree;
  entry.best_internal = std::move(best_internal);
  (void)depth;
  return entry;
}
static inline CacheEntry make_cache_entry_when_tree_is_not_provided(
  double loss,
  double pmax,
  int depth,
  std::optional<double> current_leaf_prob_on_path,
  bool is_leaf
) {
  bool is_internal = (!is_leaf);
  CacheEntry entry;
  entry.best_loss = loss;
  entry.pmax_best = pmax;
  entry.is_root_internal = is_internal;
  entry.current_leaf_prob_on_path = current_leaf_prob_on_path;
  (void)depth;
  return entry;
}
static inline void cache_subtree_solutions(
    const TreePtr& tree,
    const PathKey& path,
    int depth,
    int n,
    std::optional<double> current_leaf_prob_on_path) {
  (void)n;
  if (!tree) {
    return;
  }
  if (tree->is_leaf) {
    double obj = tree->objective;
    double pmax = tree->pred_prob;
    CacheKey key{path, depth, budget_bucket(current_leaf_prob_on_path.value_or(1.0))};
    auto cached = cache_get(key);
    if (!cached.has_value() || obj < cached->best_loss) {
      cache_set(key, make_cache_entry(tree, obj, pmax, depth, current_leaf_prob_on_path));
    }
    return;
  }
  double obj = tree->objective;
  double pmax = get_max_prob(tree);
  CacheKey key{path, depth, budget_bucket(current_leaf_prob_on_path.value_or(1.0))};
  auto cached = cache_get(key);
  if (!cached.has_value() || obj < cached->best_loss) {
    cache_set(key, make_cache_entry(tree, obj, pmax, depth, current_leaf_prob_on_path));
  }
  PathKey left_path = path_add_pair(path, tree->feature, '0');
  PathKey right_path = path_add_pair(path, tree->feature, '1');
  std::optional<double> left_current = current_leaf_prob_on_path;
  std::optional<double> right_current = current_leaf_prob_on_path;
  if (tree->left->is_leaf) {
    if (!right_current.has_value()) {
      right_current = tree->left->pred_prob;
    } else {
      right_current = std::min(right_current.value(), tree->left->pred_prob);
    }
  }
  if (tree->right->is_leaf) {
    if (!left_current.has_value()) {
      left_current = tree->right->pred_prob;
    } else {
      left_current = std::min(left_current.value(), tree->right->pred_prob);
    }
  }
  cache_subtree_solutions(tree->left, left_path, depth - 1, n, left_current);
  cache_subtree_solutions(tree->right, right_path, depth - 1, n, right_current);
}

// ------------------------ Trie Structure ------------------------
struct TrieNode;

struct Trie {
  double budget = 0.0;
  std::vector<std::tuple<double, size_t, std::shared_ptr<TrieNode>>> children_heap;
  std::unordered_map<int, std::shared_ptr<TrieNode>> children_by_feature;
  std::shared_ptr<TrieNode> leaf_node;
  size_t tie_breaker = 0;
  std::unordered_map<PathKey, TreePtr, PathKeyHash, PathKeyEq> solution_cache;

  explicit Trie(double budget_in) : budget(budget_in) {}

  void add_new_tree_to_trie(const TreePtr& tree);
  TreePtr get_best_tree_from_trie() const;
  int count_all_trees(std::optional<double> budget_in = std::nullopt) const;
  TreePtr return_solution_to_subproblem(const PathKey& subproblem_path);
};

struct TrieNode {
  int feature = -1;
  std::shared_ptr<Trie> left;
  std::shared_ptr<Trie> right;
  double min_objective = 0.0;
  bool is_leaf = false;
  double pred_prob = 0.5;
  std::vector<double> list_of_objectives;

  TrieNode(int feat, std::shared_ptr<Trie> l, std::shared_ptr<Trie> r, double obj, bool leaf, double prob)
      : feature(feat), left(std::move(l)), right(std::move(r)), min_objective(obj), is_leaf(leaf), pred_prob(prob) {
    list_of_objectives.push_back(obj);
  }
};

struct TrieHeapCmp {
  bool operator()(const std::tuple<double, size_t, std::shared_ptr<TrieNode>>& a,
                  const std::tuple<double, size_t, std::shared_ptr<TrieNode>>& b) const {
    if (std::get<0>(a) != std::get<0>(b)) {
      return std::get<0>(a) > std::get<0>(b);
    }
    return std::get<1>(a) > std::get<1>(b);
  }
};

void Trie::add_new_tree_to_trie(const TreePtr& tree) {
  if (!tree) {
    return;
  }
  if (tree->is_leaf) {
    if (!leaf_node || tree->objective < leaf_node->min_objective) {
      leaf_node = std::make_shared<TrieNode>(-1, nullptr, nullptr, tree->objective, true, tree->pred_prob);
      children_heap.emplace_back(tree->objective, tie_breaker++, leaf_node);
      std::push_heap(children_heap.begin(), children_heap.end(), TrieHeapCmp{});
    }
    return;
  }
  auto it = children_by_feature.find(tree->feature);
  if (it != children_by_feature.end()) {
    auto& child_node = it->second;
    if (tree->objective < child_node->min_objective) {
      child_node->min_objective = tree->objective;
      child_node->list_of_objectives.push_back(tree->objective);
    }
    child_node->left->add_new_tree_to_trie(tree->left);
    child_node->right->add_new_tree_to_trie(tree->right);
  } else {
    auto new_node = std::make_shared<TrieNode>(
        tree->feature,
        std::make_shared<Trie>(budget),
        std::make_shared<Trie>(budget),
        tree->objective,
        false,
        0.5);
    new_node->left->add_new_tree_to_trie(tree->left);
    new_node->right->add_new_tree_to_trie(tree->right);
    children_by_feature[tree->feature] = new_node;
    children_heap.emplace_back(tree->objective, tie_breaker++, new_node);
    std::push_heap(children_heap.begin(), children_heap.end(), TrieHeapCmp{});
  }
}

TreePtr Trie::get_best_tree_from_trie() const {
  if (children_heap.empty()) {
    return nullptr;
  }
  auto best_node = std::get<2>(children_heap.front());
  if (best_node->is_leaf) {
    return make_leaf(best_node->pred_prob, best_node->min_objective);
  }
  TreePtr left_tree = best_node->left ? best_node->left->get_best_tree_from_trie() : nullptr;
  TreePtr right_tree = best_node->right ? best_node->right->get_best_tree_from_trie() : nullptr;
  if (!left_tree || !right_tree) {
    return nullptr;
  }
  return make_node(best_node->feature, left_tree, right_tree, best_node->min_objective);
}

int Trie::count_all_trees(std::optional<double> budget_in) const {
  int count = 0;
  if (leaf_node) {
    if (!budget_in.has_value() || leaf_node->min_objective <= budget_in.value()) {
      count += 1;
    }
  }
  for (const auto& kv : children_by_feature) {
    const auto& node = kv.second;
    if (!budget_in.has_value()) {
      count += static_cast<int>(node->list_of_objectives.empty() ? 1 : node->list_of_objectives.size());
    } else {
      for (double obj : node->list_of_objectives) {
        if (obj <= budget_in.value()) {
          count += 1;
        }
      }
    }
  }
  return count;
}

TreePtr Trie::return_solution_to_subproblem(const PathKey& subproblem_path) {
  auto it = solution_cache.find(subproblem_path);
  if (it != solution_cache.end()) {
    return it->second;
  }
  TreePtr result = nullptr;
  if (subproblem_path.empty()) {
    result = get_best_tree_from_trie();
  } else {
    for (const auto& p : subproblem_path) {
      int feature = p.first;
      char direction = p.second;
      auto child_it = children_by_feature.find(feature);
      if (child_it == children_by_feature.end()) {
        continue;
      }
      auto child_node = child_it->second;
      PathKey remaining = path_remove_pair(subproblem_path, feature, direction);
      if (direction == '0') {
        if (child_node->left) {
          auto candidate = child_node->left->return_solution_to_subproblem(remaining);
          if (candidate) {
            result = candidate;
            break;
          }
        }
      } else {
        if (child_node->right) {
          auto candidate = child_node->right->return_solution_to_subproblem(remaining);
          if (candidate) {
            result = candidate;
            break;
          }
        }
      }
    }
  }
  solution_cache[subproblem_path] = result;
  return result;
}

// ------------------------ Core Algorithms ------------------------
// Feature-major layout: column `j` is contiguous, so a split reads one
// packed column instead of chasing X[row][feature] across heap rows.
struct Matrix {
  int nrows = 0;
  int ncols = 0;
  std::vector<uint8_t> cols;

  int n() const { return nrows; }
  int m() const { return ncols; }
  bool empty() const { return nrows == 0; }
  const uint8_t* column(int feature) const {
    return cols.data() + static_cast<size_t>(feature) * static_cast<size_t>(nrows);
  }

  static Matrix from_rows(const std::vector<std::vector<int>>& rows) {
    Matrix out;
    if (rows.empty()) {
      return out;
    }
    out.nrows = static_cast<int>(rows.size());
    out.ncols = static_cast<int>(rows[0].size());
    out.cols.assign(static_cast<size_t>(out.nrows) * static_cast<size_t>(out.ncols), 0);
    for (int i = 0; i < out.nrows; ++i) {
      for (int j = 0; j < out.ncols; ++j) {
        out.cols[static_cast<size_t>(j) * out.nrows + i] = rows[i][j] == 0 ? 0 : 1;
      }
    }
    return out;
  }

  static Matrix from_row_major(const double* data, int n, int m) {
    Matrix out;
    out.nrows = n;
    out.ncols = m;
    out.cols.assign(static_cast<size_t>(n) * static_cast<size_t>(m), 0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        out.cols[static_cast<size_t>(j) * n + i] = data[static_cast<size_t>(i) * m + j] == 0.0 ? 0 : 1;
      }
    }
    return out;
  }
};

struct Options {
  bool has_branching_cost = false;
  double branching_cost = 0.0;
  double min_support = 0.05;
  double cache_bucket_size = 1e-3;
  double falling_constraint_tau = 0.0;
  bool recurse_on_lower_prob_first = true;
  bool run_third_recursion = true;
};

struct SplitCandidate {
  double left_loss;
  double right_loss;
  int feature;
  std::vector<int> left_idx;
  std::vector<int> right_idx;
};

struct Partition {
  std::vector<int> left_idx;
  std::vector<int> right_idx;
  int left_pos = 0;
  int right_pos = 0;
};

// One pass: split the row subset and count positives on each side.
static inline Partition partition_feature(
    const Matrix& X,
    const std::vector<int>& y,
    const std::vector<int>& row_idx,
    int feature) {
  Partition out;
  out.left_idx.reserve(row_idx.size());
  out.right_idx.reserve(row_idx.size());
  const uint8_t* col = X.column(feature);
  for (int idx : row_idx) {
    if (col[idx] == 0) {
      out.left_idx.push_back(idx);
      out.left_pos += y[idx];
    } else {
      out.right_idx.push_back(idx);
      out.right_pos += y[idx];
    }
  }
  return out;
}

static inline std::pair<double, double> leaf_from_counts(int pos, int count, double lam, int n) {
  if (count == 0) {
    return {0.0, 0.0};
  }
  int neg = count - pos;
  double loss = std::min(static_cast<double>(pos) / static_cast<double>(n),
                         static_cast<double>(neg) / static_cast<double>(n));
  loss += lam;
  double prob = static_cast<double>(pos) / static_cast<double>(count);
  return {loss, prob};
}

static inline bool path_uses_feature(const PathKey& path, int feature) {
  for (uint8_t i = 0; i < path.size_; ++i) {
    if (path.pairs[i].first == feature) {
      return true;
    }
  }
  return false;
}

static inline double mean_y(const std::vector<int>& y, const std::vector<int>& idx) {
  if (idx.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (int i : idx) {
    sum += y[i];
  }
  return sum / static_cast<double>(idx.size());
}

// Combine two child solutions under a parent split on `feature`. The branching
// cost is added only if BOTH children are internal (non-leaf), matching the
// Combine algorithm definition.
// Packed row subsets. A split is a word-wise AND against a feature mask,
// instead of gathering one byte per selected row.
struct BitCols {
  int n = 0;
  int nwords = 0;
  int m = 0;
  std::vector<uint64_t> cols;
  std::vector<uint64_t> pos;

  const uint64_t* feature(int j) const {
    return cols.data() + static_cast<size_t>(j) * static_cast<size_t>(nwords);
  }

  static BitCols build(const Matrix& X, const std::vector<int>& y) {
    BitCols out;
    out.n = X.n();
    out.m = X.m();
    out.nwords = (out.n + 63) >> 6;
    out.cols.assign(static_cast<size_t>(out.m) * static_cast<size_t>(std::max(out.nwords, 0)), 0);
    out.pos.assign(static_cast<size_t>(std::max(out.nwords, 0)), 0);
    for (int j = 0; j < out.m; ++j) {
      const uint8_t* col = X.column(j);
      uint64_t* dst = out.cols.data() + static_cast<size_t>(j) * static_cast<size_t>(out.nwords);
      for (int i = 0; i < out.n; ++i) {
        if (col[i]) {
          dst[i >> 6] |= uint64_t{1} << (i & 63);
        }
      }
    }
    for (int i = 0; i < out.n; ++i) {
      if (y[i]) {
        out.pos[static_cast<size_t>(i >> 6)] |= uint64_t{1} << (i & 63);
      }
    }
    return out;
  }
};

struct RowSet {
  const uint64_t* bits = nullptr;
  int count = 0;
};

struct WordScratch {
  struct Chunk {
    std::unique_ptr<uint64_t[]> data;
    size_t cap = 0;
    size_t used = 0;
  };
  std::vector<Chunk> chunks;
  size_t active = 0;

  struct Mark {
    size_t active = 0;
    size_t used = 0;
  };

  Mark mark() const {
    if (chunks.empty()) {
      return {};
    }
    return {active, chunks[active].used};
  }

  void rewind(Mark m) {
    if (chunks.empty()) {
      active = 0;
      return;
    }
    active = std::min(m.active, chunks.size() - 1);
    chunks[active].used = std::min(m.used, chunks[active].cap);
    for (size_t i = active + 1; i < chunks.size(); ++i) {
      chunks[i].used = 0;
    }
  }

  void reset() { rewind({}); }

  uint64_t* alloc(size_t nwords) {
    if (nwords == 0) {
      return nullptr;
    }
    if (chunks.empty()) {
      chunks.push_back(Chunk{std::make_unique<uint64_t[]>(std::max(nwords, size_t{1 << 16})),
                             std::max(nwords, size_t{1 << 16}), 0});
      active = 0;
    }
    while (chunks[active].used + nwords > chunks[active].cap) {
      ++active;
      if (active == chunks.size()) {
        const size_t cap = std::max(nwords, size_t{1 << 16});
        chunks.push_back(Chunk{std::make_unique<uint64_t[]>(cap), cap, 0});
      } else {
        chunks[active].used = 0;
      }
    }
    uint64_t* p = chunks[active].data.get() + chunks[active].used;
    chunks[active].used += nwords;
    return p;
  }
};

static WordScratch g_word_scratch;

struct WordScope {
  WordScratch::Mark saved;
  WordScope() : saved(g_word_scratch.mark()) {}
  ~WordScope() { g_word_scratch.rewind(saved); }
};

struct BitPart {
  int left_count = 0;
  int right_count = 0;
  int left_pos = 0;
  int right_pos = 0;
};

static inline BitPart split_rows(
    const BitCols& bits, const RowSet& rows, int feature, uint64_t* left, uint64_t* right) {
  BitPart out;
  const uint64_t* feat = bits.feature(feature);
  const uint64_t* pos = bits.pos.data();
  for (int w = 0; w < bits.nwords; ++w) {
    const uint64_t keep = rows.bits[w];
    const uint64_t lb = keep & ~feat[w];
    const uint64_t rb = keep & feat[w];
    left[w] = lb;
    right[w] = rb;
    out.left_count += __builtin_popcountll(lb);
    out.right_count += __builtin_popcountll(rb);
    out.left_pos += __builtin_popcountll(lb & pos[w]);
    out.right_pos += __builtin_popcountll(rb & pos[w]);
  }
  return out;
}

static inline std::pair<double, double> leaf_from_rows(
    const BitCols& bits, const RowSet& rows, double lam, int n) {
  if (rows.count == 0) {
    return {0.0, 0.0};
  }
  int pos = 0;
  for (int w = 0; w < bits.nwords; ++w) {
    pos += __builtin_popcountll(rows.bits[w] & bits.pos[w]);
  }
  return leaf_from_counts(pos, rows.count, lam, n);
}

struct PreparedRows {
  BitCols bits;
  std::vector<uint64_t> storage;
  RowSet rows;
};

static PreparedRows prepare_rows(const Matrix& X, const std::vector<int>& y) {
  PreparedRows prepared;
  prepared.bits = BitCols::build(X, y);
  const int nwords = std::max(prepared.bits.nwords, 1);
  prepared.storage.assign(static_cast<size_t>(nwords), 0);
  for (int i = 0; i < prepared.bits.n; ++i) {
    prepared.storage[static_cast<size_t>(i >> 6)] |= uint64_t{1} << (i & 63);
  }
  prepared.rows = RowSet{prepared.storage.data(), prepared.bits.n};
  g_word_scratch.reset();
  return prepared;
}

static inline Solution combine_solutions(
    int feature,
    const Solution& s_left,
    const Solution& s_right,
    double branching_cost) {
  double L = s_left.loss + s_right.loss;
  const bool both_internal = !s_left.tree->is_leaf && !s_right.tree->is_leaf;
  if (both_internal) {
    L += branching_cost;
  }
  Solution out;
  out.loss = L;
  out.tree = make_node(feature, s_left.tree, s_right.tree, L);
  out.pmax = std::max(s_left.pmax, s_right.pmax);
  return out;
}

static OptResult OptFallingTree(
    const BitCols& bits,
    const RowSet& rows,
    int d,
    double lam,
    const std::vector<int>& features,
    int n,
    PathKey path,
    bool enable_falling_constraint,
    double current_leaf_prob_on_path,
    const Options& options,
    bool use_current_leaf_prob) {
  WordScope word_scope;
  double min_support = 0.0;
  double branching_cost = 0.0;
  if (options.has_branching_cost) {
    lam = 0.0;
    branching_cost = options.branching_cost;
    min_support = options.min_support;
  }
  g_budget_bucket_size = options.cache_bucket_size;
  if (!use_current_leaf_prob) {
    current_leaf_prob_on_path = 1.0;
  }
  // p^+_recent: the upper bound on leaf probabilities passed down from the
  // parent. Used to enforce the falling constraint along the current path.
  const double p_recent = current_leaf_prob_on_path;

  // Baseline leaf solution S^* for this subproblem.
  auto leaf_result = leaf_from_rows(bits, rows, lam, n);
  Solution leaf_sol{leaf_result.first,
                    make_leaf(leaf_result.second, leaf_result.first),
                    leaf_result.second};

  OptResult result;
  result.best = leaf_sol;
  // result.best_internal remains nullopt (S^diamond = bot).

  const int cache_budget_bucket = budget_bucket(p_recent);
  CacheKey cache_key{path, d, cache_budget_bucket};

  // Entry-level memoization: if we've already solved this exact
  // (path, depth, budget-bucket) subproblem and the cached entry carries a
  // tree, return it directly. The cache key bucketing matches the existing
  // pruning-side cache use, so any approximation tolerated there is also
  // tolerated here.
  if (auto cached = cache_get(cache_key); cached.has_value() && cached->tree) {
    OptResult cached_result;
    cached_result.best = Solution{cached->best_loss, cached->tree, cached->pmax_best};
    cached_result.best_internal = cached->best_internal;
    return cached_result;
  }

  if (d == 0) {
    cache_set(cache_key, make_cache_entry(result.best.tree, result.best.loss,
                                          result.best.pmax, d, p_recent));
    return result;
  }

  // Build candidate set C: filter out splits violating min-support or the
  // falling constraint max(p^+_L, p^+_R) <= p^+_recent.
  struct LocalCandidate {
    int feature;
    RowSet left_rows;
    RowSet right_rows;
    double left_leaf_prob;
    double right_leaf_prob;
    double leaf_sum;
  };
  std::vector<LocalCandidate> candidates;
  for (int j : features) {
    if (path_uses_feature(path, j)) {
      continue;
    }
    uint64_t* left_bits = g_word_scratch.alloc(static_cast<size_t>(bits.nwords));
    uint64_t* right_bits = g_word_scratch.alloc(static_cast<size_t>(bits.nwords));
    BitPart part = split_rows(bits, rows, j, left_bits, right_bits);
    if (static_cast<double>(part.left_count) / n <= min_support ||
        static_cast<double>(part.right_count) / n <= min_support) {
      continue;
    }
    const auto left_lc = leaf_from_counts(part.left_pos, part.left_count, lam, n);
    const auto right_lc = leaf_from_counts(part.right_pos, part.right_count, lam, n);
    const double left_prob = left_lc.second;
    const double right_prob = right_lc.second;
    const double max_prob_for_subproblem = std::max(left_prob, right_prob);
    if (enable_falling_constraint &&
        max_prob_for_subproblem > options.falling_constraint_tau &&
        max_prob_for_subproblem > p_recent) {
      continue;
    }
    candidates.push_back({j, RowSet{left_bits, part.left_count}, RowSet{right_bits, part.right_count},
                          left_prob, right_prob, left_lc.first + right_lc.first});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const LocalCandidate& a, const LocalCandidate& b) {
              return a.leaf_sum < b.leaf_sum;
            });

  // Inter-sibling falling check: when one side is a leaf and the other is
  // internal, the internal side's max p^+ must be <= the leaf's prob.
  auto check_falling_between_siblings = [&](const Solution& s_h,
                                            const Solution& s_l) -> bool {
    if (!enable_falling_constraint) {
      return true;
    }
    if (s_h.tree->is_leaf && !s_l.tree->is_leaf) {
      return s_l.pmax <= s_h.tree->pred_prob;
    }
    if (!s_h.tree->is_leaf && s_l.tree->is_leaf) {
      return s_h.pmax <= s_l.tree->pred_prob;
    }
    return true;
  };

  for (const auto& cand : candidates) {
    int j = cand.feature;
    PathKey path_L = path_add_pair(path, j, '0');
    PathKey path_R = path_add_pair(path, j, '1');

    // Assign (I_h, P_h) to the higher-p^+ child and (I_l, P_l) to the lower.
    const bool right_is_high = cand.right_leaf_prob >= cand.left_leaf_prob;
    const RowSet I_h = right_is_high ? cand.right_rows : cand.left_rows;
    const RowSet I_l = right_is_high ? cand.left_rows : cand.right_rows;
    const PathKey& P_h = right_is_high ? path_R : path_L;
    const PathKey& P_l = right_is_high ? path_L : path_R;

    // Cache-based pruning: compute a lower bound on L_split using cached
    // best losses for the (P_h, P_l) subproblems at the current budget.
    CacheKey high_key{P_h, d - 1, cache_budget_bucket};
    CacheKey low_key{P_l, d - 1, cache_budget_bucket};
    auto high_cached = cache_get(high_key);
    auto low_cached = cache_get(low_key);
    if (high_cached.has_value() && low_cached.has_value()) {
      double L_split_lower = high_cached->best_loss + low_cached->best_loss;
      if (high_cached->is_root_internal && low_cached->is_root_internal) {
        L_split_lower += branching_cost;
      }
      if (L_split_lower >= result.best.loss) {
        continue;
      }
    }

    // Recurse on the higher-p^+ child with budget p^+_recent.
    OptResult opt_h = OptFallingTree(
        bits, I_h, d - 1, lam, features, n, P_h,
        enable_falling_constraint, p_recent, options, use_current_leaf_prob);

    std::optional<Solution> S_j;  // S_j = bot

    auto combine_for_node = [&](const Solution& s_h, const Solution& s_l) -> Solution {
      const Solution& s_left = right_is_high ? s_l : s_h;
      const Solution& s_right = right_is_high ? s_h : s_l;
      return combine_solutions(j, s_left, s_right, branching_cost);
    };

    auto try_update_Sj = [&](const Solution& s) {
      // Global pmax check: combined solution must still satisfy p^+_recent.
      if (enable_falling_constraint && use_current_leaf_prob &&
          s.pmax > options.falling_constraint_tau && s.pmax > p_recent) {
        return;
      }
      if (!S_j.has_value() || s.loss < S_j->loss) {
        S_j = s;
      }
    };

    if (opt_h.best.tree->is_leaf) {
      // |> A: leaf at high, low under tightened p^+_h
      const double p_plus_h = opt_h.best.pmax;
      if (p_plus_h <= p_recent) {
        OptResult opt_l_A = OptFallingTree(
            bits, I_l, d - 1, lam, features, n, P_l,
            enable_falling_constraint, p_plus_h, options, use_current_leaf_prob);
        Solution combined = combine_for_node(opt_h.best, opt_l_A.best);
        if (check_falling_between_siblings(opt_h.best, opt_l_A.best)) {
          try_update_Sj(combined);
        }
      }
      // |> B: best non-leaf at high, low under p^+_recent
      if (opt_h.best_internal.has_value() &&
          opt_h.best_internal->pmax <= p_recent) {
        OptResult opt_l_B = OptFallingTree(
            bits, I_l, d - 1, lam, features, n, P_l,
            enable_falling_constraint, p_recent, options, use_current_leaf_prob);
        Solution combined = combine_for_node(opt_h.best_internal.value(),
                                             opt_l_B.best);
        if (check_falling_between_siblings(opt_h.best_internal.value(),
                                           opt_l_B.best)) {
          try_update_Sj(combined);
        }
      }
    } else {
      // T_h is internal: recurse on the low side under p^+_recent.
      OptResult opt_l = OptFallingTree(
          bits, I_l, d - 1, lam, features, n, P_l,
          enable_falling_constraint, p_recent, options, use_current_leaf_prob);

      if (opt_l.best.tree->is_leaf) {
        // |> A': leaf at low, re-solve high under tightened p^+_l
        const double p_plus_l = opt_l.best.pmax;
        if (p_plus_l <= p_recent) {
          OptResult opt_h_prime = OptFallingTree(
              bits, I_h, d - 1, lam, features, n, P_h,
              enable_falling_constraint, p_plus_l, options,
              use_current_leaf_prob);
          Solution combined = combine_for_node(opt_h_prime.best, opt_l.best);
          if (check_falling_between_siblings(opt_h_prime.best, opt_l.best)) {
            try_update_Sj(combined);
          }
        }
        // |> B': keep T_h, use T_l^diamond (no extra recursion)
        if (opt_l.best_internal.has_value()) {
          Solution combined = combine_for_node(opt_h.best,
                                               opt_l.best_internal.value());
          if (check_falling_between_siblings(opt_h.best,
                                             opt_l.best_internal.value())) {
            try_update_Sj(combined);
          }
        }
      } else {
        // Both children non-leaf: straightforward Combine.
        Solution combined = combine_for_node(opt_h.best, opt_l.best);
        try_update_Sj(combined);
      }
    }

    if (S_j.has_value()) {
      if (S_j->loss < result.best.loss) {
        result.best = S_j.value();
      }
      // S_j is always a Node (non-leaf), so it is a candidate for S^diamond.
      if (!result.best_internal.has_value() ||
          S_j->loss < result.best_internal->loss) {
        result.best_internal = S_j.value();
      }
    }
  }

  cache_set(cache_key,
            make_cache_entry(result.best.tree, result.best.loss,
                             result.best.pmax, d, p_recent,
                             result.best_internal));
  return result;
}

struct RSetEntry {
  TreePtr tree;
  double obj;
  double pmax;
};

// Pointer-identity pair key. R_L / R_R already contain distinct Tree
// instances (each is constructed by a unique make_node/make_leaf call), so
// pointer-based dedup is sufficient and orders of magnitude cheaper than
// the previous structural hash/equality (which recursed via tree_equal on
// every comparison).
struct PairKey {
  const Tree* a;
  const Tree* b;
  bool operator==(const PairKey& o) const noexcept { return a == o.a && b == o.b; }
};

struct PairKeyHash {
  size_t operator()(const PairKey& p) const noexcept {
    const uintptr_t pa = reinterpret_cast<uintptr_t>(p.a);
    const uintptr_t pb = reinterpret_cast<uintptr_t>(p.b);
    size_t h = std::hash<uintptr_t>{}(pa);
    h ^= std::hash<uintptr_t>{}(pb) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

struct TreePtrHash {
  size_t operator()(const TreePtr& t) const noexcept {
    return tree_hash(t);
  }
};

struct TreePtrEq {
  bool operator()(const TreePtr& a, const TreePtr& b) const noexcept {
    return tree_equal(a, b);
  }
};

static std::vector<RSetEntry> OptFallingRset(
    const BitCols& bits,
    const RowSet& rows,
    int d,
    double lam,
    double B,
    const std::vector<int>& features,
    int n,
    PathKey path,
    bool enable_falling_constraint,
    double current_leaf_prob_on_path,
    bool use_heap,
    const Options& options,
    bool use_current_leaf_prob) {
  WordScope word_scope;
  double min_support = 0.0;
  double branching_cost = 0.0;
  if (options.has_branching_cost) {
    lam = 0.0;
    branching_cost = options.branching_cost;
    min_support = options.min_support;
  }
  g_budget_bucket_size = options.cache_bucket_size;
  if (!use_current_leaf_prob) {
    current_leaf_prob_on_path = 1.0;
  }

  CacheKey cache_key{path, d, budget_bucket(current_leaf_prob_on_path)};
  auto cached_value = cache_get(cache_key);
  if (cached_value.has_value() && cached_value->best_loss > B) {
    return {};
  }

  std::vector<std::tuple<double, size_t, TreePtr, double>> heap;
  size_t tie_breaker = 0;
  std::vector<RSetEntry> list;

  auto leaf_result = leaf_from_rows(bits, rows, lam, n);
  if (leaf_result.first <= B) {
    if (use_heap) {
      heap.emplace_back(leaf_result.first, tie_breaker++, make_leaf(leaf_result.second, leaf_result.first),
                        leaf_result.second);
      std::push_heap(heap.begin(), heap.end(),
                     [](const auto& a, const auto& b) {
                       if (std::get<0>(a) != std::get<0>(b)) {
                         return std::get<0>(a) > std::get<0>(b);
                       }
                       return std::get<1>(a) > std::get<1>(b);
                     });
    } else {
      list.push_back({make_leaf(leaf_result.second, leaf_result.first), leaf_result.first, leaf_result.second});
    }
  }

  if (d == 0) {
    if (static_cast<double>(rows.count) / n <= min_support) {
      return {};
    }
    if (use_heap) {
      std::vector<std::tuple<double, size_t, TreePtr, double>> sorted_heap = heap;
      std::sort(sorted_heap.begin(), sorted_heap.end(),
                [](const auto& a, const auto& b) {
                  if (std::get<0>(a) != std::get<0>(b)) {
                    return std::get<0>(a) < std::get<0>(b);
                  }
                  return std::get<1>(a) < std::get<1>(b);
                });
      std::vector<RSetEntry> result;
      result.reserve(sorted_heap.size());
      for (const auto& item : sorted_heap) {
        result.push_back({std::get<2>(item), std::get<0>(item), std::get<3>(item)});
      }
      return result;
    }
    std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.obj < b.obj; });
    return list;
  }

  struct RsetCandidate {
    double leaf_loss;
    double left_leaf_cost;
    double right_leaf_cost;
    double left_leaf_prob;
    double right_leaf_prob;
    int feature;
    RowSet left_rows;
    RowSet right_rows;
    PathKey new_path_left;
    PathKey new_path_right;
  };

  std::vector<RsetCandidate> candidates;
  for (int j : features) {
    if (path_uses_feature(path, j)) {
      continue;
    }
    uint64_t* left_bits = g_word_scratch.alloc(static_cast<size_t>(bits.nwords));
    uint64_t* right_bits = g_word_scratch.alloc(static_cast<size_t>(bits.nwords));
    BitPart part = split_rows(bits, rows, j, left_bits, right_bits);
    if (static_cast<double>(part.left_count) / n <= min_support ||
        static_cast<double>(part.right_count) / n <= min_support) {
      continue;
    }
    const auto left_lc = leaf_from_counts(part.left_pos, part.left_count, lam, n);
    const auto right_lc = leaf_from_counts(part.right_pos, part.right_count, lam, n);
    const double max_prob_for_subproblem = std::max(left_lc.second, right_lc.second);
    if (enable_falling_constraint &&
        max_prob_for_subproblem > options.falling_constraint_tau &&
        max_prob_for_subproblem > current_leaf_prob_on_path) {
      continue;
    }
    PathKey new_path_left = path_add_pair(path, j, '0');
    PathKey new_path_right = path_add_pair(path, j, '1');
    candidates.push_back({
        left_lc.first + right_lc.first,
        left_lc.first, right_lc.first,
        left_lc.second, right_lc.second,
        j, RowSet{left_bits, part.left_count}, RowSet{right_bits, part.right_count},
        std::move(new_path_left), std::move(new_path_right)});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const RsetCandidate& a, const RsetCandidate& b) { return a.leaf_loss < b.leaf_loss; });

  double original_current_leaf_prob_on_path = current_leaf_prob_on_path;
  for (const auto& cand : candidates) {
    const int current_bucket = budget_bucket(current_leaf_prob_on_path);
    CacheKey left_key{cand.new_path_left, d - 1, current_bucket};
    CacheKey right_key{cand.new_path_right, d - 1, current_bucket};
    auto left_cached = cache_get(left_key);
    auto right_cached = cache_get(right_key);
    double L_L_star = 0.0;
    double L_R_star = 0.0;
    double pmax_L_star = 0.0;
    double pmax_R_star = 0.0;
    bool is_left_leaf = false;
    bool is_right_leaf = false;
    if (left_cached.has_value() && right_cached.has_value()) {
      L_L_star = left_cached->best_loss;
      L_R_star = right_cached->best_loss;
      pmax_L_star = left_cached->pmax_best;
      pmax_R_star = right_cached->pmax_best;
      is_left_leaf = !left_cached->is_root_internal;
      is_right_leaf = !right_cached->is_root_internal;
    } else {
      OptResult opt_L = OptFallingTree(
          bits, cand.left_rows, d - 1, lam, features, n, cand.new_path_left,
          enable_falling_constraint, current_leaf_prob_on_path, options,
          use_current_leaf_prob);
      OptResult opt_R = OptFallingTree(
          bits, cand.right_rows, d - 1, lam, features, n, cand.new_path_right,
          enable_falling_constraint, current_leaf_prob_on_path, options,
          use_current_leaf_prob);
      L_L_star = opt_L.best.loss;
      L_R_star = opt_R.best.loss;
      pmax_L_star = opt_L.best.pmax;
      pmax_R_star = opt_R.best.pmax;
      is_left_leaf = opt_L.best.tree->is_leaf;
      is_right_leaf = opt_R.best.tree->is_leaf;
      cache_set(left_key,
                make_cache_entry(opt_L.best.tree, L_L_star, std::nullopt, d - 1,
                                 std::nullopt, opt_L.best_internal));
      cache_set(right_key,
                make_cache_entry(opt_R.best.tree, L_R_star, std::nullopt, d - 1,
                                 std::nullopt, opt_R.best_internal));
    }
    double L_split = L_L_star + L_R_star;
    if (!is_left_leaf && !is_right_leaf) {
      L_split += branching_cost;
    }
    // Reuse the leaf cost / prob computed once at candidate-construction time.
    const double left_leaf_cost = cand.left_leaf_cost;
    const double right_leaf_cost = cand.right_leaf_cost;
    const double left_leaf_prob = cand.left_leaf_prob;
    const double right_leaf_prob = cand.right_leaf_prob;
    const double replace_left = left_leaf_cost + L_R_star;
    const double replace_right = right_leaf_cost + L_L_star;
    const double replace_both = left_leaf_cost + right_leaf_cost;
    const double best_replace = std::min(replace_left, std::min(replace_right, replace_both));
    const double tol = 1e-9;
    if (best_replace < L_split) {

      L_split = best_replace;
      double L_L_star_leaf = L_L_star;
      double L_R_star_leaf = L_R_star;
      double pmax_L_star_leaf = pmax_L_star;
      double pmax_R_star_leaf = pmax_R_star;
      bool is_left_leaf_for_cache = is_left_leaf;
      if (std::abs(best_replace - replace_both) <= tol) {
        L_L_star_leaf = left_leaf_cost;
        L_R_star_leaf = right_leaf_cost;
        pmax_L_star_leaf = left_leaf_prob;
        pmax_R_star_leaf = right_leaf_prob;
        is_left_leaf_for_cache = true;
      } else if (std::abs(best_replace - replace_left) <= tol) {
        L_L_star_leaf = left_leaf_cost;
        pmax_L_star_leaf = left_leaf_prob;
        is_left_leaf_for_cache = true;
      } else {
        L_R_star_leaf = right_leaf_cost;
        pmax_R_star_leaf = right_leaf_prob;
      }
      // cache_set(left_key,
      //           make_cache_entry_when_tree_is_not_provided(L_L_star_leaf, pmax_L_star_leaf, d - 1,
      //                                                      current_leaf_prob_on_path, is_left_leaf_for_cache));
      // cache_set(right_key,
      //           make_cache_entry_when_tree_is_not_provided(L_R_star_leaf, pmax_R_star_leaf, d - 1,
      //                                                      current_leaf_prob_on_path, is_right_leaf_for_cache));
    }
    // print the tree t_L and t_R if len(row_idx) == n
    
    if (L_split <= B) {
      // left_leaf_prob / right_leaf_prob from the outer scope reuse the
      // candidate's precomputed values; no extra row scan here.
      const bool take_lower_first = options.recurse_on_lower_prob_first;
      const bool choose_left_first = take_lower_first
          ? (left_leaf_prob <= right_leaf_prob)
          : (left_leaf_prob > right_leaf_prob);

      const RowSet first_rows =
          choose_left_first ? cand.left_rows : cand.right_rows;
      const PathKey& first_path =
          choose_left_first ? cand.new_path_left : cand.new_path_right;
      const double first_budget =
          choose_left_first ? (B - L_R_star) : (B - L_L_star);

      const RowSet second_rows =
          choose_left_first ? cand.right_rows : cand.left_rows;
      const PathKey& second_path =
          choose_left_first ? cand.new_path_right : cand.new_path_left;
      const double second_budget =
          choose_left_first ? (B - L_L_star) : (B - L_R_star);

      const auto is_single_leaf_rset = [](const std::vector<RSetEntry>& rset) {
        return rset.size() == 1 && rset.front().tree->is_leaf;
      };

      auto R_first = OptFallingRset(
          bits, first_rows, d - 1, lam, first_budget, features, n,
          first_path, enable_falling_constraint, original_current_leaf_prob_on_path, use_heap, options,
          use_current_leaf_prob);

      const bool first_is_single_leaf =
          use_current_leaf_prob && is_single_leaf_rset(R_first);
      const bool first_has_multiple_trees = R_first.size() > 1;
      const double second_leaf_budget = first_is_single_leaf
          ? R_first.front().tree->pred_prob
          : original_current_leaf_prob_on_path;

      auto R_second = OptFallingRset(
          bits, second_rows, d - 1, lam, second_budget, features, n,
          second_path, enable_falling_constraint, second_leaf_budget, use_heap, options,
          use_current_leaf_prob);
      std::vector<RSetEntry> R_L, R_R;
      if (choose_left_first) {
        R_L = std::move(R_first);
        R_R = std::move(R_second);
      } else {
        R_L = std::move(R_second);
        R_R = std::move(R_first);
      }
      const double min_objR = R_R.empty() ? std::numeric_limits<double>::infinity() : R_R.front().obj;
      for (const auto& left_entry : R_L) {
        if (left_entry.obj + min_objR > B) {
          break;
        }
        const bool left_leaf = left_entry.tree->is_leaf;
        const double left_prob = left_entry.tree->pred_prob;
        const double left_pmax = left_entry.pmax;
        for (const auto& right_entry : R_R) {
          double total_obj = left_entry.obj + right_entry.obj;
          const bool right_leaf = right_entry.tree->is_leaf;
          if (!left_leaf && !right_leaf) {
            total_obj += branching_cost;
          }
          if (total_obj > B) {
            break;
          }
          if (left_leaf && right_leaf) {
            const double right_prob = right_entry.tree->pred_prob;
            if (left_prob < 0.5 && right_prob < 0.5) {
              continue;
            }
            if (left_prob > 0.5 && right_prob > 0.5) {
              continue;
            }
          }
          const double cand_pmax = std::max(left_pmax, right_entry.pmax);
          if (enable_falling_constraint && cand_pmax > options.falling_constraint_tau) {
            if (left_leaf && !right_leaf) {
              if (left_pmax < right_entry.pmax) {
                continue;
              }
            } else if (!left_leaf && right_leaf) {
              if (right_entry.pmax < left_pmax) {
                continue;
              }
            }
            if (use_current_leaf_prob && cand_pmax > original_current_leaf_prob_on_path) {
              continue;
            }
          }
          TreePtr t_candidate = make_node(cand.feature, left_entry.tree, right_entry.tree, total_obj);
          if (use_heap) {
            heap.emplace_back(total_obj, tie_breaker++, t_candidate, cand_pmax);
            std::push_heap(heap.begin(), heap.end(),
                  [](const auto& a, const auto& b) {
                    if (std::get<0>(a) != std::get<0>(b)) {
                      return std::get<0>(a) > std::get<0>(b);
                    }
                    return std::get<1>(a) > std::get<1>(b);
                  });
          } else {
            list.push_back({t_candidate, total_obj, cand_pmax});
          }
        }
      }
    }
  }

  std::vector<RSetEntry> result;
  if (use_heap) {
    std::vector<std::tuple<double, size_t, TreePtr, double>> sorted_heap = heap;
    std::sort(sorted_heap.begin(), sorted_heap.end(),
              [](const auto& a, const auto& b) {
                if (std::get<0>(a) != std::get<0>(b)) {
                  return std::get<0>(a) < std::get<0>(b);
                }
                return std::get<1>(a) < std::get<1>(b);
              });
    result.reserve(sorted_heap.size());
    for (const auto& item : sorted_heap) {
      result.push_back({std::get<2>(item), std::get<0>(item), std::get<3>(item)});
    }
  } else {
    std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.obj < b.obj; });
    result = std::move(list);
  }
  return result;
}

// ------------------------ Pretty Printing ------------------------
static void print_tree(const TreePtr& t, int depth = 0) {
  std::string ind(depth * 2, ' ');
  if (!t) {
    std::cout << ind << "None\n";
    return;
  }
  if (t->is_leaf) {
    std::cout << ind << "Leaf(pred_prob=" << std::fixed << std::setprecision(3) << t->pred_prob
              << ", objective=" << t->objective << ")\n";
  } else {
    std::cout << ind << "Node(feature=" << t->feature << ")\n";
    print_tree(t->left, depth + 1);
    print_tree(t->right, depth + 1);
  }
}

static std::string tree_to_string(const TreePtr& t, int depth = 0) {
  std::ostringstream oss;
  std::string ind(depth * 2, ' ');
  if (!t) {
    oss << ind << "None\n";
    return oss.str();
  }
  if (t->is_leaf) {
    oss << ind << "Leaf(pred_prob=" << std::fixed << std::setprecision(3) << t->pred_prob
        << ", objective=" << t->objective << ")\n";
  } else {
    oss << ind << "Node(feature=" << t->feature << ")\n";
    oss << tree_to_string(t->left, depth + 1);
    oss << tree_to_string(t->right, depth + 1);
  }
  return oss.str();
}

// ------------------------ Serialization ------------------------
static std::string tree_to_json(const TreePtr& t) {
  if (!t) {
    return "null";
  }
  std::ostringstream oss;
  oss << std::setprecision(17);
  if (t->is_leaf) {
    oss << "{\"t\":\"L\",\"p\":" << t->pred_prob << ",\"o\":" << t->objective << "}";
  } else {
    oss << "{\"t\":\"N\",\"f\":" << t->feature << ",\"o\":" << t->objective
        << ",\"l\":" << tree_to_json(t->left)
        << ",\"r\":" << tree_to_json(t->right) << "}";
  }
  return oss.str();
}

static bool dump_rset_to_jsonl(const std::vector<RSetEntry>& R, const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) {
    return false;
  }
  out << std::setprecision(17);
  for (const auto& entry : R) {
    out << "{\"obj\":" << entry.obj << ",\"pmax\":" << entry.pmax
        << ",\"tree\":" << tree_to_json(entry.tree) << "}\n";
  }
  return true;
}

// ------------------------ CSV Loading ------------------------
static std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> parts;
  std::string token;
  std::stringstream ss(line);
  while (std::getline(ss, token, ',')) {
    parts.push_back(token);
  }
  if (!line.empty() && line.back() == ',') {
    parts.push_back("");
  }
  return parts;
}

static bool load_compas_binarized_csv(
    const std::string& path,
    Matrix& X,
    std::vector<int>& y,
    std::vector<std::string>& feature_cols,
    std::string& label_col) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return false;
  }
  std::string line;
  if (!std::getline(in, line)) {
    return false;
  }
  auto header = split_csv_line(line);
  if (header.size() < 2) {
    return false;
  }
  feature_cols.assign(header.begin(), header.end() - 1);
  label_col = header.back();
  std::vector<std::vector<int>> rows;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    auto parts = split_csv_line(line);
    if (parts.size() != header.size()) {
      continue;
    }
    std::vector<int> row;
    row.reserve(parts.size() - 1);
    bool parse_ok = true;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
      try {
        int v = std::stoi(parts[i]);
        row.push_back(v);
      } catch (...) {
        parse_ok = false;
        break;
      }
    }
    if (!parse_ok) {
      continue;
    }
    int label = 0;
    try {
      label = std::stoi(parts.back());
    } catch (...) {
      continue;
    }
    rows.push_back(std::move(row));
    y.push_back(label);
  }
  X = Matrix::from_rows(rows);
  return !X.empty() && X.n() == static_cast<int>(y.size());
}

#ifndef FALLING_TREES_PYBIND
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: falling_rashomon <path_to_binarized_csv> [--disable_path_prob]\n";
    return 1;
  }
  const std::string csv_path = argv[1];
  bool use_current_leaf_prob = true;
  for (int i = 2; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "--disable_path_prob") {
      use_current_leaf_prob = false;
    }
  }
  Matrix X;
  std::vector<int> y;
  std::vector<std::string> feature_cols;
  std::string label_col;
  std::cout << "Loading Compas binarized dataset...\n";
  if (!load_compas_binarized_csv(csv_path, X, y, feature_cols, label_col)) {
    std::cerr << "Failed to load dataset from: " << csv_path << "\n";
    return 1;
  }

  int n = X.n();
  int m = X.m();
  std::cout << "Dataset shape: " << n << " samples, " << m << " features\n";
  std::cout << "Feature columns: [";
  for (size_t i = 0; i < feature_cols.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << feature_cols[i];
  }
  std::cout << "]\n";
  std::cout << "Label column: " << label_col << "\n";
  double pos_prop_all = 0.0;
  for (int v : y) {
    pos_prop_all += v;
  }
  pos_prop_all /= static_cast<double>(y.size());
  std::cout << "Positive label proportion: " << std::fixed << std::setprecision(3) << pos_prop_all << "\n";

  std::vector<int> features(m);
  std::iota(features.begin(), features.end(), 0);

  double lam = 0.005;
  double eps = 0.02;
  int depth = 5;
  bool enable_falling_constraint = true;
  bool use_heap = true;
  MAX_CACHE_SIZE = 1000000;
  cache_clear();

  Options options;
  options.has_branching_cost = true;
  options.branching_cost = 0.01;
  options.min_support = 0.02;

  PreparedRows prepared = prepare_rows(X, y);

  std::cout << "\nRunning OptFallingTree on Compas dataset (depth=" << depth << ")...\n";
  auto start_time = std::chrono::high_resolution_clock::now();
  OptResult opt_top = OptFallingTree(
      prepared.bits, prepared.rows, depth, lam, features, n, PathKey{},
      enable_falling_constraint, 1.0, options, use_current_leaf_prob);
  const double best_loss = opt_top.best.loss;
  const TreePtr best_tree = opt_top.best.tree;
  const double pmax = opt_top.best.pmax;
  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

  bool is_best_tree_falling = is_falling_tree(best_tree);
  bool is_best_tree_falling_new = is_falling_tree_new_constraint(best_tree);
  std::cout << "Is best tree falling?: " << (is_best_tree_falling ? "True" : "False") << "\n";
  std::cout << "Is best tree falling (new constraint)?: " << (is_best_tree_falling_new ? "True" : "False") << "\n";
  std::cout << "Best loss: " << std::fixed << std::setprecision(4) << best_loss
            << ", pmax: " << pmax << "\n";
  std::cout << "Time elapsed: " << std::fixed << std::setprecision(2) << elapsed_time << " seconds\n";
  std::cout << "\nBest tree:\n";
  print_tree(best_tree);
  std::cout << "Normalized colless index of best tree: "
            << std::fixed << std::setprecision(4) << normalized_colless_index(best_tree) << "\n";
  std::cout << "Subproblem memo size: " << _subproblem_cache.size() << "\n";

  std::cout << "\nComputing R-set with budget = best_loss (should include at least the best tree):\n";
  
  double B = 0.3366;
  start_time = std::chrono::high_resolution_clock::now();
  auto R = OptFallingRset(
      prepared.bits, prepared.rows, depth, lam, B, features, n, PathKey{}, enable_falling_constraint,
      1.0, use_heap, options, use_current_leaf_prob);
  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
  std::cout << "Subproblem memo size after OptFallingRset: " << _subproblem_cache.size() << "\n";
  std::cout << "Found " << R.size() << " trees in R-set with budget <= "
            << std::fixed << std::setprecision(4) << B << "\n";
  std::cout << "Time elapsed: " << std::fixed << std::setprecision(2) << elapsed_time << " seconds\n";

  std::vector<double> colless_values;
  colless_values.reserve(R.size());
  int non_falling_new = 0;
  for (const auto& entry : R) {
    colless_values.push_back(normalized_colless_index(entry.tree));
    if (!is_falling_tree_new_constraint(entry.tree)) {
      non_falling_new += 1;
    }
  }
  if (non_falling_new > 0) {
    std::cout << "WARNING: " << non_falling_new << " tree(s) violate the new falling constraint.\n";
  } else {
    std::cout << "All trees in R-set satisfy the new falling constraint.\n";
  }
  if (!colless_values.empty()) {
    double sum = std::accumulate(colless_values.begin(), colless_values.end(), 0.0);
    double avg = sum / static_cast<double>(colless_values.size());
    auto minmax = std::minmax_element(colless_values.begin(), colless_values.end());
    std::cout << "Average normalized colless index: " << std::fixed << std::setprecision(4) << avg << "\n";
    std::cout << "Min normalized colless index: " << std::fixed << std::setprecision(4) << *minmax.first << "\n";
    std::cout << "Max normalized colless index: " << std::fixed << std::setprecision(4) << *minmax.second << "\n";
  } else {
    std::cout << "Average normalized colless index: 0.0000\n";
    std::cout << "Min normalized colless index: 0.0000\n";
    std::cout << "Max normalized colless index: 0.0000\n";
  }
  std::unordered_set<TreePtr, TreePtrHash, TreePtrEq> unique_trees;
  unique_trees.reserve(R.size());
  for (const auto& entry : R) {
    unique_trees.insert(entry.tree);
  }
  std::cout << "Unique trees in R-set (structural): " << unique_trees.size()
            << " (duplicates: " << (R.size() - unique_trees.size()) << ")\n";
  std::cout << "Cache hits: " << _cache_hits << ", cache misses: " << _cache_misses << "\n";

  const std::string rset_dump_path =
      "/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/falling_trees/cpp_rset_trees.jsonl";
  if (dump_rset_to_jsonl(R, rset_dump_path)) {
    std::cout << "Saved C++ R-set to: " << rset_dump_path << "\n";
  } else {
    std::cout << "WARNING: failed to save R-set to: " << rset_dump_path << "\n";
  }
  return 0;
}
#endif

#ifdef FALLING_TREES_PYBIND
namespace py = pybind11;

static int predict_tree_row_from_json(
    const py::dict& node,
    const int* row,
    ssize_t n_features,
    double threshold) {
  const std::string node_type = node["t"].cast<std::string>();
  if (node_type == "L") {
    const double pred_prob = node["p"].cast<double>();
    return pred_prob >= threshold ? 1 : 0;
  }
  if (node_type != "N") {
    throw std::runtime_error("Invalid node type in tree JSON.");
  }
  const int feature = node["f"].cast<int>();
  if (feature < 0 || feature >= n_features) {
    throw std::runtime_error("Tree feature index out of bounds for input data.");
  }
  const int value = row[feature];
  const py::object next_node = value == 0 ? node["l"] : node["r"];
  return predict_tree_row_from_json(py::cast<py::dict>(next_node), row, n_features, threshold);
}

static py::array_t<int> predict_tree_from_json(
    const py::dict& tree,
    py::array X_in,
    double threshold) {
  py::array_t<int, py::array::c_style | py::array::forcecast> X_arr = X_in;
  if (X_arr.ndim() != 2) {
    throw std::runtime_error("X must be a 2D array.");
  }
  const ssize_t n = X_arr.shape(0);
  const ssize_t m = X_arr.shape(1);
  auto X = X_arr.unchecked<2>();
  py::array_t<int> preds(n);
  auto preds_mut = preds.mutable_unchecked<1>();
  for (ssize_t i = 0; i < n; ++i) {
    preds_mut(i) = predict_tree_row_from_json(tree, &X(i, 0), m, threshold);
  }
  return preds;
}

static py::array_t<int> predict(
    py::dict tree,
    py::array X_in,
    double threshold) {
  return predict_tree_from_json(tree, X_in, threshold);
}

static py::array_t<int> predict_json(
    const std::string& tree_json,
    py::array X_in,
    double threshold) {
  py::object json = py::module_::import("json");
  py::object tree_obj = json.attr("loads")(tree_json);
  return predict_tree_from_json(py::cast<py::dict>(tree_obj), X_in, threshold);
}

static py::dict run_from_csv(
    const std::string& csv_path,
    double lam,
    double eps,
    int depth,
    bool enable_falling_constraint,
    bool use_heap,
    double branching_cost,
    double min_support,
    bool use_current_leaf_prob,
    double cache_bucket_size,
    double falling_constraint_tau,
    bool recurse_on_lower_prob_first,
    bool run_third_recursion,
    std::optional<double> budget_override,
    std::optional<std::string> dump_rset_jsonl,
    size_t max_cache_size) {
  Matrix X;
  std::vector<int> y;
  std::vector<std::string> feature_cols;
  std::string label_col;
  if (!load_compas_binarized_csv(csv_path, X, y, feature_cols, label_col)) {
    throw std::runtime_error("Failed to load dataset: " + csv_path);
  }
  int n = X.n();
  int m = X.m();
  std::vector<int> features(m);
  std::iota(features.begin(), features.end(), 0);
  PreparedRows prepared = prepare_rows(X, y);

  Options options;
  options.has_branching_cost = true;
  options.branching_cost = branching_cost;
  options.min_support = min_support;
  options.cache_bucket_size = cache_bucket_size;
  options.falling_constraint_tau = falling_constraint_tau;
  options.recurse_on_lower_prob_first = recurse_on_lower_prob_first;
  options.run_third_recursion = run_third_recursion;

  MAX_CACHE_SIZE = max_cache_size;
  cache_clear();

  OptResult opt_top = OptFallingTree(
      prepared.bits, prepared.rows, depth, lam, features, n, PathKey{},
      enable_falling_constraint, 1.0, options, use_current_leaf_prob);
  const double best_loss = opt_top.best.loss;
  const TreePtr best_tree = opt_top.best.tree;
  const double pmax = opt_top.best.pmax;

  double B = budget_override.has_value() ? budget_override.value() : best_loss * (1.0 + eps);
  auto R = OptFallingRset(
      prepared.bits, prepared.rows, depth, lam, B, features, n, PathKey{}, enable_falling_constraint,
      1.0, use_heap, options, use_current_leaf_prob);

  int non_falling_new = 0;
  for (const auto& entry : R) {
    if (!is_falling_tree_new_constraint(entry.tree)) {
      non_falling_new += 1;
    }
  }

  if (dump_rset_jsonl.has_value()) {
    if (!dump_rset_to_jsonl(R, dump_rset_jsonl.value())) {
      throw std::runtime_error("Failed to dump R-set JSONL: " + dump_rset_jsonl.value());
    }
  }

  py::dict result;
  result["best_loss"] = best_loss;
  result["pmax"] = pmax;
  result["rset_size"] = static_cast<size_t>(R.size());
  result["best_tree_str"] = tree_to_string(best_tree);
  result["cache_size"] = static_cast<size_t>(_subproblem_cache.size());
  result["cache_hits"] = static_cast<size_t>(_cache_hits);
  result["cache_misses"] = static_cast<size_t>(_cache_misses);
  result["non_falling_new"] = non_falling_new;
  return result;
}

static py::dict run_from_dataframe(
    py::object df,
    double lam,
    double eps,
    int depth,
    bool enable_falling_constraint,
    bool use_heap,
    double branching_cost,
    double min_support,
    bool use_current_leaf_prob,
    double cache_bucket_size,
    double falling_constraint_tau,
    bool recurse_on_lower_prob_first,
    bool run_third_recursion,
    std::optional<double> budget_override,
    std::optional<std::string> dump_rset_jsonl,
    size_t max_cache_size) {
  py::object np = py::module_::import("numpy");
  py::object arr_obj = df.attr("to_numpy")();
  py::array arr = py::array(arr_obj);
  if (arr.ndim() != 2) {
    throw std::runtime_error("DataFrame must be 2D");
  }
  const ssize_t n = arr.shape(0);
  const ssize_t m = arr.shape(1);
  if (m < 2) {
    throw std::runtime_error("DataFrame must have at least 2 columns");
  }
  py::array_t<double, py::array::c_style | py::array::forcecast> arr_d = arr;
  auto buf = arr_d.request();
  const double* data = static_cast<double*>(buf.ptr);

  const int n_features = static_cast<int>(m - 1);
  Matrix X;
  X.nrows = static_cast<int>(n);
  X.ncols = n_features;
  X.cols.assign(static_cast<size_t>(n) * static_cast<size_t>(n_features), 0);
  std::vector<int> y(static_cast<size_t>(n));
  for (ssize_t i = 0; i < n; ++i) {
    for (ssize_t j = 0; j < n_features; ++j) {
      X.cols[static_cast<size_t>(j) * n + i] = data[i * m + j] == 0.0 ? 0 : 1;
    }
    y[static_cast<size_t>(i)] = static_cast<int>(data[i * m + (m - 1)]);
  }

  std::vector<int> features(static_cast<size_t>(m - 1));
  std::iota(features.begin(), features.end(), 0);
  PreparedRows prepared = prepare_rows(X, y);

  Options options;
  options.has_branching_cost = true;
  options.branching_cost = branching_cost;
  options.min_support = min_support;
  options.cache_bucket_size = cache_bucket_size;
  options.falling_constraint_tau = falling_constraint_tau;
  options.recurse_on_lower_prob_first = recurse_on_lower_prob_first;
  options.run_third_recursion = run_third_recursion;

  MAX_CACHE_SIZE = max_cache_size;
  cache_clear();

  OptResult opt_top = OptFallingTree(
      prepared.bits, prepared.rows, depth, lam, features, static_cast<int>(n), PathKey{},
      enable_falling_constraint, 1.0, options, use_current_leaf_prob);
  const double best_loss = opt_top.best.loss;
  const TreePtr best_tree = opt_top.best.tree;
  const double pmax = opt_top.best.pmax;

  double B = budget_override.has_value() ? budget_override.value() : best_loss * (1.0 + eps);
  auto R = OptFallingRset(
      prepared.bits, prepared.rows, depth, lam, B, features, static_cast<int>(n), PathKey{}, enable_falling_constraint,
      1.0, use_heap, options, use_current_leaf_prob);

  int non_falling_new = 0;
  for (const auto& entry : R) {
    if (!is_falling_tree_new_constraint(entry.tree)) {
      non_falling_new += 1;
    }
  }

  if (dump_rset_jsonl.has_value()) {
    if (!dump_rset_to_jsonl(R, dump_rset_jsonl.value())) {
      throw std::runtime_error("Failed to dump R-set JSONL: " + dump_rset_jsonl.value());
    }
  }

  py::dict result;
  result["best_loss"] = best_loss;
  result["pmax"] = pmax;
  result["rset_size"] = static_cast<size_t>(R.size());
  result["best_tree_str"] = tree_to_string(best_tree);
  result["cache_size"] = static_cast<size_t>(_subproblem_cache.size());
  result["cache_hits"] = static_cast<size_t>(_cache_hits);
  result["cache_misses"] = static_cast<size_t>(_cache_misses);
  result["non_falling_new"] = non_falling_new;
  return result;
}

static py::dict run_from_xy(
    py::array X_in,
    py::array y_in,
    double lam,
    double eps,
    int depth,
    bool enable_falling_constraint,
    bool use_heap,
    double branching_cost,
    double min_support,
    bool use_current_leaf_prob,
    double cache_bucket_size,
    double falling_constraint_tau,
    bool recurse_on_lower_prob_first,
    bool run_third_recursion,
    std::optional<double> budget_override,
    std::optional<std::string> dump_rset_jsonl,
    size_t max_cache_size) {
  py::array_t<double, py::array::c_style | py::array::forcecast> X_arr = X_in;
  py::array_t<double, py::array::c_style | py::array::forcecast> y_arr = y_in;
  if (X_arr.ndim() != 2) {
    throw std::runtime_error("X must be 2D");
  }
  if (y_arr.ndim() != 1) {
    throw std::runtime_error("y must be 1D");
  }
  const ssize_t n = X_arr.shape(0);
  const ssize_t m = X_arr.shape(1);
  if (y_arr.shape(0) != n) {
    throw std::runtime_error("X and y must have the same number of rows");
  }
  auto X_buf = X_arr.request();
  auto y_buf = y_arr.request();
  const double* X_data = static_cast<double*>(X_buf.ptr);
  const double* y_data = static_cast<double*>(y_buf.ptr);

  Matrix X = Matrix::from_row_major(X_data, static_cast<int>(n), static_cast<int>(m));
  std::vector<int> y(static_cast<size_t>(n));
  for (ssize_t i = 0; i < n; ++i) {
    y[static_cast<size_t>(i)] = static_cast<int>(y_data[i]);
  }

  std::vector<int> features(static_cast<size_t>(m));
  std::iota(features.begin(), features.end(), 0);
  PreparedRows prepared = prepare_rows(X, y);

  Options options;
  options.has_branching_cost = true;
  options.branching_cost = branching_cost;
  options.min_support = min_support;
  options.cache_bucket_size = cache_bucket_size;
  options.falling_constraint_tau = falling_constraint_tau;
  options.recurse_on_lower_prob_first = recurse_on_lower_prob_first;
  options.run_third_recursion = run_third_recursion;

  MAX_CACHE_SIZE = max_cache_size;
  cache_clear();

  OptResult opt_top = OptFallingTree(
      prepared.bits, prepared.rows, depth, lam, features, static_cast<int>(n), PathKey{},
      enable_falling_constraint, 1.0, options, use_current_leaf_prob);
  const double best_loss = opt_top.best.loss;
  const TreePtr best_tree = opt_top.best.tree;
  const double pmax = opt_top.best.pmax;

  double B = budget_override.has_value() ? budget_override.value() : best_loss * (1.0 + eps);
  auto R = OptFallingRset(
      prepared.bits, prepared.rows, depth, lam, B, features, static_cast<int>(n), PathKey{}, enable_falling_constraint,
      1.0, use_heap, options, use_current_leaf_prob);
  // add best tree to R (in front of the list)
  R.insert(R.begin(), {best_tree, best_loss, pmax});

  int non_falling_new = 0;
  for (const auto& entry : R) {
    if (!is_falling_tree_new_constraint(entry.tree)) {
      non_falling_new += 1;
    }
  }

  if (dump_rset_jsonl.has_value()) {
    if (!dump_rset_to_jsonl(R, dump_rset_jsonl.value())) {
      throw std::runtime_error("Failed to dump R-set JSONL: " + dump_rset_jsonl.value());
    }
  }

  py::dict result;
  result["best_loss"] = best_loss;
  result["pmax"] = pmax;
  result["rset_size"] = static_cast<size_t>(R.size());
  result["best_tree_str"] = tree_to_string(best_tree);
  result["cache_size"] = static_cast<size_t>(_subproblem_cache.size());
  result["cache_hits"] = static_cast<size_t>(_cache_hits);
  result["cache_misses"] = static_cast<size_t>(_cache_misses);
  result["non_falling_new"] = non_falling_new;
  return result;
}

PYBIND11_MODULE(falling_rashomon_cpp, m) {
  m.doc() = "Pybind11 wrapper for falling tree Rashomon set C++ implementation";
  m.def("predict", &predict,
        py::arg("tree"),
        py::arg("X"),
        py::arg("threshold") = 0.5);
  m.def("predict_json", &predict_json,
        py::arg("tree_json"),
        py::arg("X"),
        py::arg("threshold") = 0.5);
  m.def("run_from_csv", &run_from_csv,
        py::arg("csv_path"),
        py::arg("lam") = 0.005,
        py::arg("eps") = 0.02,
        py::arg("depth") = 5,
        py::arg("enable_falling_constraint") = true,
        py::arg("use_heap") = true,
        py::arg("branching_cost") = 0.01,
        py::arg("min_support") = 0.02,
        py::arg("use_current_leaf_prob") = true,
        py::arg("cache_bucket_size") = 1e-3,
        py::arg("falling_constraint_tau") = 0.0,
        py::arg("recurse_on_lower_prob_first") = true,
        py::arg("run_third_recursion") = true,
        py::arg("budget_override") = py::none(),
        py::arg("dump_rset_jsonl") = py::none(),
        py::arg("max_cache_size") = 1000000);
  m.def("run_from_dataframe", &run_from_dataframe,
        py::arg("df"),
        py::arg("lam") = 0.005,
        py::arg("eps") = 0.02,
        py::arg("depth") = 5,
        py::arg("enable_falling_constraint") = true,
        py::arg("use_heap") = true,
        py::arg("branching_cost") = 0.01,
        py::arg("min_support") = 0.02,
        py::arg("use_current_leaf_prob") = true,
        py::arg("cache_bucket_size") = 1e-3,
        py::arg("falling_constraint_tau") = 0.0,
        py::arg("recurse_on_lower_prob_first") = true,
        py::arg("run_third_recursion") = true,
        py::arg("budget_override") = py::none(),
        py::arg("dump_rset_jsonl") = py::none(),
        py::arg("max_cache_size") = 1000000);
  m.def("run_from_xy", &run_from_xy,
        py::arg("X"),
        py::arg("y"),
        py::arg("lam") = 0.005,
        py::arg("eps") = 0.02,
        py::arg("depth") = 5,
        py::arg("enable_falling_constraint") = true,
        py::arg("use_heap") = true,
        py::arg("branching_cost") = 0.01,
        py::arg("min_support") = 0.02,
        py::arg("use_current_leaf_prob") = true,
        py::arg("cache_bucket_size") = 1e-3,
        py::arg("falling_constraint_tau") = 0.0,
        py::arg("recurse_on_lower_prob_first") = true,
        py::arg("run_third_recursion") = true,
        py::arg("budget_override") = py::none(),
        py::arg("dump_rset_jsonl") = py::none(),
        py::arg("max_cache_size") = 1000000);
}
#endif

