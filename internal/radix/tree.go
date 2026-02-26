package radix

import (
	"container/list"
	"sync"
)

// Tree represents a Radix prefix tree for KV cache management
type Tree struct {
	// Root is the empty root node (no tokens, handle=0)
	Root *Node

	// mu protects concurrent access to tree structure
	// Readers use RLock, writers use Lock
	mu sync.RWMutex

	// lruList is the intrusive doubly-linked list for O(1) eviction
	// Only contains unpinned leaf nodes ready for eviction
	lruList *list.List
}

// NewTree creates an empty Radix tree with initialized root
func NewTree() *Tree {
	return &Tree{
		Root:    NewNode(nil, nil), // Root has no tokens
		lruList: list.New(),
	}
}

// Match finds the longest prefix match for given tokens
// Returns the deepest node whose token sequence is a prefix of query
// Returns nil if no match found
// Thread-safe: uses RLock for concurrent reads
func (t *Tree) Match(tokens []uint32) *Node {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return t.match(tokens)
}

// match is the internal implementation without locking
func (t *Tree) match(tokens []uint32) *Node {
	current := t.Root
	var bestMatch *Node = nil

	// Root always matches (empty prefix)
	if len(tokens) == 0 {
		return current
	}

	for {
		// Find child whose first token matches current position
		firstToken := tokens[0]
		child, exists := current.Children[firstToken]
		if !exists {
			// No more matching children
			break
		}

		// Check how many tokens match between query and child edge
		common := longestCommonPrefix(tokens, child.Tokens)

		if common == 0 {
			// No match at this position
			break
		}

		if common == len(child.Tokens) {
			// Entire child edge is a prefix of query
			// This node is a valid match
			if child.IsReady() {
				bestMatch = child
			}

			// Continue deeper with remaining tokens
			tokens = tokens[common:]
			current = child

			// If we've consumed all tokens, we're done
			if len(tokens) == 0 {
				break
			}
			continue
		}

		// Query is a prefix of child edge (common == len(tokens))
		// This is a valid match - we can use this node's cache as base
		if common == len(tokens) && child.IsReady() {
			bestMatch = child
		}

		// Partial match with remaining tokens in both - diverging path
		break
	}

	return bestMatch
}

// longestCommonPrefix returns the length of the longest common prefix
// of two token sequences
func longestCommonPrefix(a, b []uint32) int {
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	for i := 0; i < minLen; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return minLen
}
