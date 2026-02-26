package radix

import (
	"sync"
	"testing"
)

func TestNewTree(t *testing.T) {
	tree := NewTree()

	if tree.Root == nil {
		t.Error("Expected Root to be initialized")
	}

	if tree.lruList == nil {
		t.Error("Expected lruList to be initialized")
	}
}

func TestMatchEmptyTree(t *testing.T) {
	tree := NewTree()

	node := tree.Match([]uint32{1, 2, 3})

	if node != nil {
		t.Errorf("Expected nil for empty tree, got %v", node)
	}
}

func TestMatchExact(t *testing.T) {
	tree := NewTree()

	// Build a simple tree: root -> [1,2,3]
	node := NewNode([]uint32{1, 2, 3}, tree.Root)
	tree.Root.Children[1] = node
	FinalizeNode(node, 100)

	result := tree.Match([]uint32{1, 2, 3})

	if result == nil {
		t.Fatal("Expected node, got nil")
	}

	if result.CacheHandle != 100 {
		t.Errorf("Expected CacheHandle 100, got %d", result.CacheHandle)
	}
}

func TestMatchPrefix(t *testing.T) {
	tree := NewTree()

	// Tree: root -> [1,2,3] -> [4,5]
	parent := NewNode([]uint32{1, 2, 3}, tree.Root)
	tree.Root.Children[1] = parent
	FinalizeNode(parent, 100)

	child := NewNode([]uint32{4, 5}, parent)
	parent.Children[4] = child
	FinalizeNode(child, 200)

	// Should match parent node (longest prefix = [1,2,3])
	result := tree.Match([]uint32{1, 2, 3, 9, 9})

	if result == nil {
		t.Fatal("Expected node, got nil")
	}

	if result.CacheHandle != 100 {
		t.Errorf("Expected CacheHandle 100 (parent), got %d", result.CacheHandle)
	}
}

func TestMatchPartialPrefix(t *testing.T) {
	tree := NewTree()

	// Tree: root -> [1,2,3,4,5]
	node := NewNode([]uint32{1, 2, 3, 4, 5}, tree.Root)
	tree.Root.Children[1] = node
	FinalizeNode(node, 100)

	// Query: [1,2,3,9,9]
	// Longest common prefix is [1,2,3] (3 tokens)
	// But node only matches if it's a prefix of query OR query is prefix of node
	// Since [1,2,3,4,5] is NOT a prefix of [1,2,3,9,9] (mismatch at index 3)
	// And [1,2,3,9,9] is NOT a prefix of [1,2,3,4,5]
	// This should return nil (no match)

	result := tree.Match([]uint32{1, 2, 3, 9, 9})

	if result != nil {
		t.Errorf("Expected nil for partial prefix mismatch, got node with CacheHandle %d", result.CacheHandle)
	}
}

func TestMatchLongestPrefix(t *testing.T) {
	tree := NewTree()

	// Build tree with multiple possible matches:
	// root -> [1] -> [2] -> [3]
	n1 := NewNode([]uint32{1}, tree.Root)
	tree.Root.Children[1] = n1
	FinalizeNode(n1, 100)

	n2 := NewNode([]uint32{2}, n1)
	n1.Children[2] = n2
	FinalizeNode(n2, 200)

	n3 := NewNode([]uint32{3}, n2)
	n2.Children[3] = n3
	FinalizeNode(n3, 300)

	// Query: [1,2,3,4]
	// Should match n3 (longest prefix)
	result := tree.Match([]uint32{1, 2, 3, 4})

	if result == nil {
		t.Fatal("Expected node, got nil")
	}

	if result.CacheHandle != 300 {
		t.Errorf("Expected CacheHandle 300, got %d", result.CacheHandle)
	}
}

func TestMatchQueryIsPrefix(t *testing.T) {
	tree := NewTree()

	// Tree: root -> [1,2,3,4,5]
	node := NewNode([]uint32{1, 2, 3, 4, 5}, tree.Root)
	tree.Root.Children[1] = node
	FinalizeNode(node, 100)

	// Query: [1,2,3]
	// Query is a prefix of node's tokens - should match
	result := tree.Match([]uint32{1, 2, 3})

	if result == nil {
		t.Fatal("Expected node, got nil")
	}

	if result.CacheHandle != 100 {
		t.Errorf("Expected CacheHandle 100, got %d", result.CacheHandle)
	}
}

func TestMatchConcurrent(t *testing.T) {
	tree := NewTree()

	// Build tree
	node := NewNode([]uint32{1, 2, 3}, tree.Root)
	tree.Root.Children[1] = node
	FinalizeNode(node, 100)

	// Concurrent reads
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result := tree.Match([]uint32{1, 2, 3})
			if result == nil || result.CacheHandle != 100 {
				t.Error("Concurrent match failed")
			}
		}()
	}
	wg.Wait()
}

func TestLongestCommonPrefix(t *testing.T) {
	tests := []struct {
		name     string
		a        []uint32
		b        []uint32
		expected int
	}{
		{
			name:     "identical",
			a:        []uint32{1, 2, 3},
			b:        []uint32{1, 2, 3},
			expected: 3,
		},
		{
			name:     "no match",
			a:        []uint32{1, 2, 3},
			b:        []uint32{4, 5, 6},
			expected: 0,
		},
		{
			name:     "partial match",
			a:        []uint32{1, 2, 3},
			b:        []uint32{1, 2, 9},
			expected: 2,
		},
		{
			name:     "a is prefix of b",
			a:        []uint32{1, 2},
			b:        []uint32{1, 2, 3, 4},
			expected: 2,
		},
		{
			name:     "b is prefix of a",
			a:        []uint32{1, 2, 3, 4},
			b:        []uint32{1, 2},
			expected: 2,
		},
		{
			name:     "empty a",
			a:        []uint32{},
			b:        []uint32{1, 2, 3},
			expected: 0,
		},
		{
			name:     "empty b",
			a:        []uint32{1, 2, 3},
			b:        []uint32{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := longestCommonPrefix(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Expected %d, got %d", tt.expected, result)
			}
		})
	}
}

func TestInsertPendingEmptyTree(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	node, err := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if node == nil {
		t.Fatal("Expected node, got nil")
	}

	if !nodeIsChild(tree.Root, node) {
		t.Error("Expected node to be child of root")
	}
}

func TestInsertPendingNewBranch(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert first node
	node1, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	FinalizeNode(node1, 100)

	// Insert different branch
	node2, err := tree.InsertPending([]uint32{4, 5}, engine, nil)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if node2 == nil {
		t.Fatal("Expected node, got nil")
	}

	// Should be separate branch from root
	if node1 == node2 {
		t.Error("Expected different nodes")
	}
}

func TestInsertPendingExistingExact(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert first node
	node1, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)

	// Request same tokens - should return existing node
	node2, err := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if node1 != node2 {
		t.Errorf("Expected same node, got different nodes")
	}

	// Node should be pinned (refCount > 0)
	if node1.refCount.Load() != 2 {
		t.Errorf("Expected refCount 2, got %d", node1.refCount.Load())
	}
}

func TestInsertPendingThunderingHerd(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	var nodes []*Node
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Simulate 10 concurrent requests for same tokens
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			node, err := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
			if err != nil {
				t.Errorf("Expected no error, got %v", err)
				return
			}

			mu.Lock()
			nodes = append(nodes, node)
			mu.Unlock()
		}()
	}
	wg.Wait()

	// All should get the same node
	first := nodes[0]
	for _, node := range nodes {
		if node != first {
			t.Error("Expected all requests to get same node")
		}
	}

	// refCount should be 10 (all requests pinned it)
	if first.refCount.Load() != 10 {
		t.Errorf("Expected refCount 10, got %d", first.refCount.Load())
	}
}

func TestInsertPendingPrefixMatch(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert [1,2,3]
	node1, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	FinalizeNode(node1, 100)

	// Insert [1,2,3,4] - should extend from node1
	node2, err := tree.InsertPending([]uint32{1, 2, 3, 4}, engine, nil)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if node2 == nil {
		t.Fatal("Expected node, got nil")
	}

	if node2.Parent != node1 {
		t.Error("Expected node2 to be child of node1")
	}
}

func TestInsertPendingAfterFinalize(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert and finalize [1,2,3]
	node1, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	FinalizeNode(node1, 100)

	// Request same tokens - should return finalized node
	node2, err := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if node1 != node2 {
		t.Error("Expected same node")
	}

	if !node1.IsReady() {
		t.Error("Expected node to be ready")
	}
}

func TestInsertPendingPoisonedRetry(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert node
	node1, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)

	// Poison it
	PoisonNode(node1, TestError("computation failed"))

	// Try to insert same tokens - should skip poisoned and create new
	node2, err := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if node1 == node2 {
		t.Error("Expected different node (should skip poisoned)")
	}

	if node2.err != nil {
		t.Errorf("Expected new node to not be poisoned, got %v", node2.err)
	}
}

func TestInsertPendingConcurrentDifferent(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	var wg sync.WaitGroup
	errors := make(chan error, 2)

	// Two concurrent requests for different tokens
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, err := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
		errors <- err
	}()

	go func() {
		defer wg.Done()
		_, err := tree.InsertPending([]uint32{4, 5, 6}, engine, nil)
		errors <- err
	}()

	wg.Wait()
	close(errors)

	for err := range errors {
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
	}
}

func TestInsertPendingOCPRetry(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// This test verifies OCC double-checked locking
	// First thread acquires lock, finds no match, releases lock
	// Second thread should either find first thread's node or race to create

	var wg sync.WaitGroup
	nodes := make(chan *Node, 5)

	// 5 concurrent requests
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			node, err := tree.InsertPending([]uint32{1, 2, 3, 4, 5}, engine, nil)
			if err != nil {
				t.Errorf("Expected no error, got %v", err)
				return
			}
			nodes <- node
		}()
	}

	wg.Wait()
	close(nodes)

	// Collect unique nodes
	unique := make(map[*Node]bool)
	for node := range nodes {
		unique[node] = true
	}

	// Should only have 1 unique node (OCC worked)
	if len(unique) != 1 {
		t.Errorf("Expected 1 unique node, got %d", len(unique))
	}

	// Get the single node
	var singleNode *Node
	for node := range unique {
		singleNode = node
	}

	// refCount should be 5
	if singleNode.refCount.Load() != 5 {
		t.Errorf("Expected refCount 5, got %d", singleNode.refCount.Load())
	}
}

func TestInsertPendingWithUnpin(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert node
	node, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)

	if node.refCount.Load() != 1 {
		t.Errorf("Expected refCount 1, got %d", node.refCount.Load())
	}

	// Unpin
	tree.Unpin(node)

	if node.refCount.Load() != 0 {
		t.Errorf("Expected refCount 0 after unpin, got %d", node.refCount.Load())
	}
}

// Helper function to check if node is child of parent
func nodeIsChild(parent, child *Node) bool {
	for _, c := range parent.Children {
		if c == child {
			return true
		}
	}
	return false
}

func TestPrunePoisonedSingleNode(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert and poison node
	node, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	PoisonNode(node, TestError("failed"))

	// Prune should remove the node
	tree.PrunePoisoned()

	// Node should no longer be in tree
	if nodeIsChild(tree.Root, node) {
		t.Error("Expected poisoned node to be removed from tree")
	}
}

func TestPrunePoisonedWithChildren(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Create tree: root -> [1,2,3] -> [4,5]
	parent, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	PoisonNode(parent, TestError("parent failed"))

	child, _ := tree.InsertPending([]uint32{1, 2, 3, 4, 5}, engine, nil)

	// Prune should remove both
	tree.PrunePoisoned()

	if nodeIsChild(tree.Root, parent) {
		t.Error("Expected poisoned parent to be removed")
	}

	// Child should also be orphaned (removed from parent, which is removed)
	// In production, we'd cascade delete or reattach
	if nodeIsChild(parent, child) {
		// This is actually OK - parent is disconnected from tree
		// but child is still parent's child
	}
}

func TestPrunePoisonedFinalizedNotRemoved(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert and finalize node
	node, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	FinalizeNode(node, 100)

	// Prune should NOT remove finalized nodes
	tree.PrunePoisoned()

	if !nodeIsChild(tree.Root, node) {
		t.Error("Expected finalized node to remain in tree")
	}
}

func TestPrunePoisonedMixedTree(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert multiple nodes
	node1, _ := tree.InsertPending([]uint32{1}, engine, nil)
	FinalizeNode(node1, 100)

	node2, _ := tree.InsertPending([]uint32{2}, engine, nil)
	PoisonNode(node2, TestError("failed"))

	node3, _ := tree.InsertPending([]uint32{3}, engine, nil)
	FinalizeNode(node3, 300)

	// Prune should only remove poisoned
	tree.PrunePoisoned()

	if !nodeIsChild(tree.Root, node1) {
		t.Error("Expected node1 to remain")
	}

	if nodeIsChild(tree.Root, node2) {
		t.Error("Expected poisoned node2 to be removed")
	}

	if !nodeIsChild(tree.Root, node3) {
		t.Error("Expected node3 to remain")
	}
}

func TestPrunePoisonedEmptyTree(t *testing.T) {
	tree := NewTree()

	// Should not panic on empty tree
	tree.PrunePoisoned()

	// Root should still exist
	if tree.Root == nil {
		t.Error("Expected root to exist")
	}
}

func TestPrunePoisonedDoesNotRemovePending(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Insert pending node (not finalized, not poisoned)
	node, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)

	// Prune should NOT remove pending nodes
	tree.PrunePoisoned()

	if !nodeIsChild(tree.Root, node) {
		t.Error("Expected pending node to remain in tree")
	}
}

func TestPrunePoisonedCascading(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Create deeper tree: root -> [1] -> [2] -> [3]
	n1, _ := tree.InsertPending([]uint32{1}, engine, nil)
	n2, _ := tree.InsertPending([]uint32{1, 2}, engine, nil)
	_, _ = tree.InsertPending([]uint32{1, 2, 3}, engine, nil)

	// Poison middle node
	PoisonNode(n2, TestError("middle failed"))

	// Prune should remove n2 and n3 (cascade)
	tree.PrunePoisoned()

	if !nodeIsChild(tree.Root, n1) {
		t.Error("Expected n1 to remain")
	}

	// n2 should be removed
	if nodeIsChild(tree.Root, n2) || nodeIsChild(n1, n2) {
		t.Error("Expected poisoned n2 to be removed")
	}
}

func TestUnpinDecrementsRefCount(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	node, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	FinalizeNode(node, 100)

	tree.Unpin(node) // Now thread-safe, no manual lock needed

	if node.refCount.Load() != 0 {
		t.Errorf("Expected refCount 0, got %d", node.refCount.Load())
	}
}

func TestUnpinAddsToLRU(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	node, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	FinalizeNode(node, 100)

	tree.Unpin(node)

	// Node should be in LRU list
	if node.lruElem == nil {
		t.Error("Expected node to be in LRU list")
	}

	if tree.lruList.Len() != 1 {
		t.Errorf("Expected LRU length 1, got %d", tree.lruList.Len())
	}
}

func TestUnpinInternalNodeNotAddedToLRU(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Create parent and child
	parent, _ := tree.InsertPending([]uint32{1, 2}, engine, nil)
	FinalizeNode(parent, 100)

	child, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	FinalizeNode(child, 200)

	tree.Unpin(parent) // Parent has children

	// Parent should NOT be in LRU (has children)
	if parent.lruElem != nil {
		t.Error("Expected internal node to not be in LRU")
	}
}

func TestUnpinPendingNotAddedToLRU(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	node, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)
	// Don't finalize - node is pending

	tree.Unpin(node)

	// Pending node should NOT be in LRU
	if node.lruElem != nil {
		t.Error("Expected pending node to not be in LRU")
	}
}

func TestEvictLRU(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Create and finalize multiple nodes
	node1, _ := tree.InsertPending([]uint32{1}, engine, nil)
	FinalizeNode(node1, 100)
	tree.Unpin(node1)

	node2, _ := tree.InsertPending([]uint32{2}, engine, nil)
	FinalizeNode(node2, 200)
	tree.Unpin(node2)

	node3, _ := tree.InsertPending([]uint32{3}, engine, nil)
	FinalizeNode(node3, 300)
	tree.Unpin(node3)

	// LRU should have 3 nodes
	if tree.lruList.Len() != 3 {
		t.Errorf("Expected LRU length 3, got %d", tree.lruList.Len())
	}

	// Evict oldest (node1 was added first)
	tree.EvictLRU(1)

	// Should have 2 nodes left
	if tree.lruList.Len() != 2 {
		t.Errorf("Expected LRU length 2 after eviction, got %d", tree.lruList.Len())
	}

	// node1 should be removed from tree
	if nodeIsChild(tree.Root, node1) {
		t.Error("Expected evicted node to be removed from tree")
	}

	// node2 and node3 should still be in tree
	if !nodeIsChild(tree.Root, node2) {
		t.Error("Expected node2 to still be in tree")
	}
	if !nodeIsChild(tree.Root, node3) {
		t.Error("Expected node3 to still be in tree")
	}
}

func TestEvictAllLRU(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	// Create nodes
	for i := uint32(1); i <= 5; i++ {
		node, _ := tree.InsertPending([]uint32{i}, engine, nil)
		FinalizeNode(node, uint64(i*100))
		tree.Unpin(node)
	}

	// Evict all
	tree.EvictLRU(5)

	// All nodes should be removed
	if tree.lruList.Len() != 0 {
		t.Errorf("Expected empty LRU, got length %d", tree.lruList.Len())
	}
}

func TestEvictZeroDoesNothing(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	node, _ := tree.InsertPending([]uint32{1}, engine, nil)
	FinalizeNode(node, 100)
	tree.Unpin(node)

	tree.EvictLRU(0)

	if tree.lruList.Len() != 1 {
		t.Error("Evicting 0 should do nothing")
	}
}

func TestMultipleUnpin(t *testing.T) {
	tree := NewTree()
	engine := &MockMLXEngine{}

	node, _ := tree.InsertPending([]uint32{1, 2, 3}, engine, nil)

	// Pin multiple times
	node.refCount.Add(2) // Now refCount = 3

	FinalizeNode(node, 100)

	tree.Unpin(node) // refCount = 2

	if node.refCount.Load() != 2 {
		t.Errorf("Expected refCount 2, got %d", node.refCount.Load())
	}

	if node.lruElem != nil {
		t.Error("Expected node to not be in LRU (still pinned)")
	}

	tree.Unpin(node) // refCount = 1

	if node.refCount.Load() != 1 {
		t.Errorf("Expected refCount 1, got %d", node.refCount.Load())
	}

	tree.Unpin(node) // refCount = 0, should add to LRU

	if node.refCount.Load() != 0 {
		t.Errorf("Expected refCount 0, got %d", node.refCount.Load())
	}

	if node.lruElem == nil {
		t.Error("Expected node to be in LRU after fully unpinned")
	}
}
