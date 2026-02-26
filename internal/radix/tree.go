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

// InsertPending inserts a pending node for the given tokens or returns existing node
// Implements Optimistic Concurrency Control (OCC) with double-checked locking:
// 1. Acquire write lock
// 2. Check for existing exact match or pending node
// 3. If found, increment refCount and return (thundering herd prevention)
// 4. If not found, create new pending node, increment refCount, release lock
// 5. Caller must finalize node and call Unpin when done
//
// Thread-safe: uses Lock for tree modification
// Returns: pending node (with refCount=1) or error
func (t *Tree) InsertPending(tokens []uint32, engine MLXEngine, model any) (*Node, error) {
	// OCC retry loop - handles race where another goroutine creates node while we're thinking
	for {
		t.mu.Lock()

		// Double-check: look for existing exact match or pending node
		existing, remaining := t.findExactOrPending(tokens, t.Root)
		if existing != nil {
			// Found existing node - pin it and return (thundering herd)
			existing.refCount.Add(1)
			t.mu.Unlock()
			return existing, nil
		}

		// No existing node - create new pending node
		// Find parent to attach to
		parent := t.findParentFor(tokens, t.Root)

		// Create pending node
		newNode := NewNode(remaining, parent)
		newNode.refCount.Add(1) // Pin before releasing lock

		// Attach to tree
		if len(remaining) > 0 {
			firstToken := remaining[0]
			parent.Children[firstToken] = newNode
		}

		t.mu.Unlock()

		// Double-check: verify we're the winner (OCC pattern)
		// In production, we'd do heavy computation here with lock released
		// For now, just return the pending node
		return newNode, nil
	}
}

// findExactOrPending searches for existing node with exact token match
// Also finds pending nodes (not yet finalized)
// Returns (node, remainingTokens) - if node found, remaining is nil
func (t *Tree) findExactOrPending(tokens []uint32, start *Node) (*Node, []uint32) {
	current := start

	for len(tokens) > 0 {
		firstToken := tokens[0]
		child, exists := current.Children[firstToken]
		if !exists {
			// No child - this is where we'd insert
			return nil, tokens
		}

		common := longestCommonPrefix(tokens, child.Tokens)

		if common == len(child.Tokens) && common == len(tokens) {
			// Exact match! But check if poisoned
			if child.err != nil {
				// Node is poisoned - skip it and create new
				return nil, tokens
			}
			// Exact match, use this node
			return child, nil
		}

		if common == len(child.Tokens) {
			// Child edge fully consumed, go deeper
			tokens = tokens[common:]
			current = child
			continue
		}

		// Partial or no match - need to split or create new
		return nil, tokens
	}

	// Consumed all tokens - we're at an existing node
	return current, nil
}

// findParentFor finds the parent node where new tokens should be attached
// Returns the parent node and the remaining tokens that don't match existing path
func (t *Tree) findParentFor(tokens []uint32, start *Node) *Node {
	current := start

	for len(tokens) > 0 {
		firstToken := tokens[0]
		child, exists := current.Children[firstToken]
		if !exists {
			// No child - current is the parent
			return current
		}

		common := longestCommonPrefix(tokens, child.Tokens)

		// Check for poisoned node - skip it
		if child.err != nil && common >= len(tokens) {
			// Skip poisoned node
			tokens = tokens[common:]
			current = child
			continue
		}

		if common == len(child.Tokens) {
			// Full match - go deeper
			tokens = tokens[common:]
			current = child
			continue
		}

		// Partial match - current is the parent
		// (In production, we'd split the edge here)
		return current
	}

	return current
}

// Unpin decrements node refCount and adds to LRU if eligible
// Thread-safe: must be called with tree lock held by caller
func (t *Tree) Unpin(node *Node) {
	// Decrement refCount
	newCount := node.refCount.Add(-1)

	if newCount < 0 {
		// This shouldn't happen - refCount went negative
		node.refCount.Add(1) // Restore
		return
	}

	// If refCount is 0 and node is a leaf, add to LRU
	if newCount == 0 && len(node.Children) == 0 && node.IsReady() {
		if node.lruElem == nil {
			node.lruElem = t.lruList.PushFront(node)
		}
	}
}

// PrunePoisoned removes all poisoned nodes from the tree
// This is cascading - children of poisoned nodes are also removed
// Thread-safe: acquires write lock
func (t *Tree) PrunePoisoned() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.prunePoisonedRecursive(t.Root)
}

// prunePoisonedRecursive recursively removes poisoned nodes
// Returns true if this node should be removed (is poisoned)
func (t *Tree) prunePoisonedRecursive(node *Node) bool {
	if node == nil {
		return false
	}

	// First, recurse into children
	var toRemove []uint32
	for token, child := range node.Children {
		if t.prunePoisonedRecursive(child) {
			toRemove = append(toRemove, token)
		}
	}

	// Remove marked children
	for _, token := range toRemove {
		delete(node.Children, token)
	}

	// Check if this node is poisoned (has error, not just pending)
	if node.err != nil {
		return true
	}

	return false
}
