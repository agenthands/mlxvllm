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
